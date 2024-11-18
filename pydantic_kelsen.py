import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from pydantic import BaseModel, Field, field_validator, model_validator
import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader, Dataset
import logging
from typing import Union, List, Dict, Tuple, Optional
import re
from difflib import SequenceMatcher
import numpy as np
import os
import argparse
import psutil
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import pydantic
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

extra_id_to_special_token = {
    "<extra_id_0>": "string",
    "<extra_id_1>": "asset",
    "<extra_id_2>": "subject",
    "<extra_id_3>": "clause",
    "<extra_id_4>": "=",
    "<extra_id_5>": ")",
    "<extra_id_6>": "CR(",
    "<extra_id_7>": "PVG(",
    "<extra_id_8>": "OB(",
    "<extra_id_9>": "PR(",
    "<extra_id_10>": "Service",
    "<extra_id_11>": "Property",
    "<extra_id_12>": "N",
    "<extra_id_13>": "NM",
    "<extra_id_14>": "+",
    "<extra_id_15>": "-",
    "<extra_id_16>": "COMPRADOR",
    "<extra_id_17>": "VENDEDOR",
    "<extra_id_18>": "PROPIETARIO",
    "<extra_id_19>": "ACREEDOR",
    "<extra_id_20>": "DEUDOR",
    "<extra_id_21>": "ADQUIRENTE",
    "<extra_id_22>": "{",
    "<extra_id_23>": "}",
    "<extra_id_24>": ";",
    "<extra_id_25>": "AND",
    "<extra_id_26>": "OFERENTE",
    "<extra_id_27>": "MUTUANTE",
    "<extra_id_28>": "MUTUARIO",
    "<extra_id_29>": "ARRENDADOR",
    "<extra_id_30>": "ARRENDATARIO",
    "<extra_id_31>":"PERMUTANTE1",
    "<extra_id_32>":"PERMUTANTE2",
    "<extra_id_33>":"DONANTE",
    "<extra_id_34>":"DONATARIO",       
    "<extra_id_35>":"PRESTADOR",
    "<extra_id_36>":"ACREDITADO",

    }


def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def log_memory_usage(step):
    memory = get_memory_usage()
    print(f"Memory usage after {step}: {memory:.2f} MB")



class StringDeclaration(BaseModel):
    identifier: str
    value: str

    @field_validator('identifier')
    @classmethod
    def validate_identifier(cls, v: str) -> str:
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', v):
            raise ValueError('Invalid identifier: must start with letter or underscore, followed by letters, numbers, or underscores')
        return v

    @field_validator('value')
    @classmethod
    def validate_value(cls, v: str) -> str:
        if not v.startswith('"') or not v.endswith('"'):
            raise ValueError('String value must be a verb enclosed in double quotes')
        return v

class SubjectDeclaration(BaseModel):
    identifier: str
    name: str
    address: str
    age: int
    email: str

    @field_validator('identifier')
    @classmethod
    def validate_identifier(cls, v: str) -> str:
        if not re.match(r'^[A-Z][A-Z0-9_]*$', v):
            raise ValueError('Subject identifier must be uppercase letters, numbers, or underscores')
        return v

    @field_validator('age')
    @classmethod
    def validate_age(cls, v: int) -> int:
        if v < 0 or v > 150:
            raise ValueError('Age must be between 0 and 150')
        return v

class AssetDeclaration(BaseModel):
    identifier: str
    asset_type: str
    subtype: str
    subject1: str
    action: str
    subject2: str

    @field_validator('asset_type')
    @classmethod
    def validate_asset_type(cls, v: str) -> str:
        if v not in ['Service', 'Property']:
            raise ValueError('Asset type must be either Service or Property')
        return v

    @field_validator('subtype')
    @classmethod
    def validate_subtype(cls, v: str, info) -> str:
        values = info.data
        if 'asset_type' in values:
            if values['asset_type'] == 'Service' and v not in ['+', '-']:
                raise ValueError('Service assets must have + or - subtype')
            elif values['asset_type'] == 'Property' and v not in ['M', 'NM']:
                raise ValueError('Property assets must have M or NM subtype')
        return v


class LegalPosition(BaseModel):
    type: str  # CR, OB, PR, PVG, PWR, LIAB, IMM, DIS
    arg1: Optional[str] = None
    arg2: Optional[str] = None
    arg3: Optional[str] = None

    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        valid_types = ['CR', 'OB', 'PR', 'PVG', 'PWR', 'LIAB', 'IMM', 'DIS']
        if v not in valid_types:
            raise ValueError(f'Invalid legal position type. Must be one of: {", ".join(valid_types)}')
        return v

class Condition(BaseModel):
    assets: List[str]
    operator: Optional[str] = None

    @field_validator('operator')
    @classmethod
    def validate_operator(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v != 'AND':
            raise ValueError('Only AND operator is supported in conditions')
        return v

class ClauseDeclaration(BaseModel):
    identifier: str
    condition: Condition
    consequence: LegalPosition

class KelsenModelOutput(BaseModel):
    strings: List[StringDeclaration] = Field(default_factory=list)
    subjects: List[SubjectDeclaration] = Field(default_factory=list)
    assets: List[AssetDeclaration] = Field(default_factory=list)
    clauses: List[ClauseDeclaration] = Field(
        default_factory=list,
        description="At least one clause is required"
    )
    @model_validator(mode='after')
    def validate_references(self) -> 'KelsenModelOutput':
        # Get all defined identifiers
        string_ids = {s.identifier for s in self.strings}
        subject_ids = {s.identifier for s in self.subjects}
        asset_ids = {a.identifier for a in self.assets}

        # Validate mandatory constructs
        validations = []
        if not self.strings:
            validations.append("At least one string declaration is required")
        if not self.assets:
            validations.append("At least one asset declaration is required")
        if not self.clauses:
            validations.append("At least one clause declaration is required")
            
        if validations:
            raise ValueError("; ".join(validations))

        # Check asset references in clauses
        for clause in self.clauses:
            for asset_ref in clause.condition.assets:
                if asset_ref not in asset_ids:
                    raise ValueError(f'Asset {asset_ref} referenced in clause but not defined')

        return self

    def to_kelsen_code(self) -> str:
        """Convert the model to formatted Kelsen code"""
        parts = []
        
        # Add strings
        for string in self.strings:
            parts.append(f'string {string.identifier} = {string.value};')
            
        # Add assets
        for asset in self.assets:
            parts.append(
                f'asset {asset.identifier} = {asset.asset_type}, {asset.subtype}, '
                f'{asset.subject1}, {asset.action}, {asset.subject2};'
            )
            
        # Add clauses
        for clause in self.clauses:
            condition_str = ' AND '.join(clause.condition.assets)
            parts.append(
                f'clause {clause.identifier} = {{\n'
                f'    {condition_str},\n'
                f'    {clause.consequence.type}({clause.consequence.arg1})\n'
                f'}};'
            )
            
        return '\n'.join(parts)
    
    def model_dump_json(self) -> str:
        """Convert to JSON string"""
        return self.model_dump()

    @classmethod
    def parse_kelsen_code(cls, code: str) -> 'KelsenModelOutput':
        """Parse Kelsen code string into structured output"""
        declarations = code.split(';')
        result = {
            'strings': [],
            'assets': [],
            'clauses': []
        }

        parsing_errors = []

        for decl in declarations:
            decl = decl.strip()
            if not decl:
                continue
                
            try:
                parts = decl.split('=', 1)
                if len(parts) != 2:
                    continue
                    
                header, content = parts[0].strip(), parts[1].strip()
                header_parts = header.split()
                
                if len(header_parts) < 2:
                    continue
                    
                decl_type, identifier = header_parts[0], header_parts[1]
                
                if decl_type == 'string':
                    result['strings'].append(StringDeclaration(
                        identifier=identifier,
                        value=content.strip()
                    ))
                elif decl_type == 'asset':
                    parts = [p.strip() for p in content.split(',')]
                    if len(parts) >= 5:  # Minimum required parts
                        result['assets'].append(AssetDeclaration(
                            identifier=identifier,
                            asset_type=parts[0],
                            subtype=parts[1],
                            subject1=parts[2],
                            action=parts[3],
                            subject2=parts[4]
                        ))
                elif decl_type == 'clause':
                    if '{' in content and '}' in content:
                        clause_content = content.strip('{}')
                        condition_part, consequence_part = clause_content.split(',', 1)
                        
                        # Parse condition
                        assets = [a.strip() for a in condition_part.split('AND')]
                        condition = Condition(
                            assets=assets,
                            operator='AND' if len(assets) > 1 else None
                        )
                        
                        # Parse consequence
                        consequence_match = re.match(r'(\w+)\((.*)\)', consequence_part.strip())
                        if consequence_match:
                            consequence = LegalPosition(
                                type=consequence_match.group(1),
                                arg1=consequence_match.group(2)
                            )
                            
                            result['clauses'].append(ClauseDeclaration(
                                identifier=identifier,
                                condition=condition,
                                consequence=consequence
                            ))
                
            except Exception as e:
                parsing_errors.append(f"Error parsing {decl}: {str(e)}")
                logger.error(f"Error parsing declaration: {decl}")
                logger.error(str(e))
        
        if parsing_errors:
            logger.warning("Parsing completed with errors: " + "; ".join(parsing_errors))
        
        return cls(**result)

    @staticmethod
    def _parse_subject_content(content: str) -> Dict:
    	"""Parse subject declaration content"""
    	# Remove surrounding brackets if present
    	content = content.strip('{}')
    	parts = content.split(',')
    	if len(parts) != 4:
    	    raise ValueError("Subject declaration must have name, address, age, and email")
    
    	return {
    	    'name': parts[0].strip(),
    	    'address': parts[1].strip(),
    	    'age': int(parts[2].strip()),
    	    'email': parts[3].strip()
    	}

    @staticmethod
    def _parse_asset_content(content: str) -> Dict:
    	"""Parse asset declaration content"""
    	parts = [p.strip() for p in content.split(',')]
    	if len(parts) != 5:
    	    raise ValueError("Asset declaration must have type, subtype, subject1, action, subject2")
    
    	return {
    	    'asset_type': parts[0],
    	    'subtype': parts[1],
    	    'subject1': parts[2],
    	    'action': parts[3],
    	    'subject2': parts[4]
    	}

    @staticmethod
    def _parse_clause_content(content: str) -> Dict:
        """Parse clause declaration content"""
        content = content.strip('{}')
        condition, consequence = content.split(',', 1)
        
        # Parse condition
        assets = [a.strip() for a in condition.split('AND')]
        condition_dict = {
            'assets': assets,
            'operator': 'AND' if len(assets) > 1 else None
        }
        
        # Parse consequence
        consequence = consequence.strip()
        match = re.match(r'(\w+)\((.*)\)', consequence)
        if not match:
            raise ValueError("Invalid consequence format")
        
        consequence_dict = {
            'type': match.group(1),
            'arg1': match.group(2)
        }
        
        return {
            'condition': condition_dict,
            'consequence': consequence_dict
        }

class ValidationErrorType(Enum):
    STRING = "string"
    SUBJECT = "subject"
    ASSET = "asset" 
    CLAUSE = "clause"
    SYNTAX = "syntax"
    TYPE = "type"

@dataclass 
class ValidationFeedback:
    is_valid: bool
    error_type: Optional[ValidationErrorType]
    error_message: Optional[str]
    error_location: Optional[str]
    suggested_correction: Optional[str]

class KelsenValidator:
    def __init__(self, model, tokenizer, max_attempts: int = 3):
        self.model = model
        self.tokenizer = tokenizer
        self.max_attempts = max_attempts
        self.valid_examples = self.generate_kelsen_examples()
        self.feedback_history = []

    def _generate_code(self, prompt: str) -> str:
        """Generate Kelsen code using the model"""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=512,
                min_length=50,
                num_beams=3,
                do_sample=True,
                temperature=0.2,
                no_repeat_ngram_size=0,
                repetition_penalty=1.0,
                early_stopping=False,
                use_cache=True,
                num_return_sequences=1
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return self.post_process_output(generated_text)

    # Alternatively, you could use this simpler approach:
    def _parse_code(self, code: str) -> Dict:
        """Parse generated code into structured format"""
        try:
            return KelsenModelOutput.parse_kelsen_code(code)
        except Exception as e:
            logger.error(f"Failed to parse Kelsen code: {str(e)}")
            # Just re-raise the original error if it's already a ValidationError
            if isinstance(e, pydantic.ValidationError):
                raise e
            # Otherwise, raise a ValueError which will be handled by the validate_and_generate method
            raise ValueError(f"Failed to parse code: {str(e)}")


    def post_process_output(self, text: str) -> str:
        """Clean up the generated text"""
        # Remove unwanted tokens
        unwanted_tokens = ['<pad>', '<s>', '</s>']
        for token in unwanted_tokens:
            text = text.replace(token, '')

        # Replace special tokens with their actual values
        for extra_id, special_token in extra_id_to_special_token.items():
            text = text.replace(extra_id, special_token)

        # Ensure proper spacing around punctuation
        text = re.sub(r'\s*([,;={}()])\s*', r' \1 ', text)
        
        # Ensure proper spacing around quotes
        text = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', text)
        
        # Handle potential artifacts from tokenization
        text = text.replace('Ġ', '')
        
        # Clean up any double spaces
        text = ' '.join(text.split())
        
        return text.strip()

    def _create_correction_prompt(self, original_text: str, error_msg: str) -> str:
            """Create a simple correction prompt for non-ValidationError cases"""
            return f"""
        Fix the following Kelsen code:

        Original code:
        {original_text}

        Error:
        {error_msg}

        Here are examples of valid structures:
        {self.valid_examples["strings"]}
        {self.valid_examples["assets"]}
        {self.valid_examples["clauses"]}

        Generate corrected code that follows these patterns:
        """

    def validate_and_generate(self, prompt: str) -> str:
        """Main entry point for code generation with validation"""
        attempts = 0
        current_prompt = prompt
        self.feedback_history = []

        while attempts < self.max_attempts:
            try:
                # Generate code
                generated_code = self._generate_code(current_prompt)
                logger.debug(f"Generated code (attempt {attempts + 1}):\n{generated_code}")
                
                # Try parsing and validating
                try:
                    parsed_output = self._parse_code(generated_code)
                    return generated_code  # Return if validation succeeds
                    
                except pydantic.ValidationError as ve:
                    error_type = self._classify_error(str(ve))
                    feedback = ValidationFeedback(
                        is_valid=False,
                        error_type=error_type,
                        error_message=str(ve),
                        error_location=str([err["loc"] for err in ve.errors()]),
                        suggested_correction=self._get_relevant_examples([str(ve)])
                    )
                    self.feedback_history.append(feedback)
                    
                    # Update prompt with correction suggestions
                    current_prompt = self._create_correction_prompt(
                        generated_code,
                        ve,
                        error_type
                    )
                    
            except Exception as e:
                logger.error(f"Unexpected error during generation: {str(e)}")
                logger.error("Error details:", exc_info=True)
                raise

            attempts += 1
            logger.info(f"Attempt {attempts}/{self.max_attempts} completed")

        raise ValueError(f"Failed to generate valid code after {self.max_attempts} attempts. "
                    f"Last errors: {[f.error_message for f in self.feedback_history]}")
   
    @staticmethod
    def generate_kelsen_examples():
        """Generate valid Kelsen examples matching parser grammar"""
        return {
            "strings": """
                # Basic string declaration
                string transferir = "transferir la propiedad";
                string pagar = "realizar el pago acordado";
            """,
            
            
            "assets": """
                # Service and Property assets
                asset Servicio = Service, +, VENDEDOR, transferir, COMPRADOR;
                asset Propiedad = Property, NM, VENDEDOR, pagar, COMPRADOR;
            """,
            
            "clauses": """
                # Simple clause with one condition
                clause simple = {
                    Servicio,
                    CR(Propiedad)
                };
                
                # Complex clause with AND condition
                clause compleja = {
                    Servicio AND Propiedad,
                    OB(Servicio)
                };
            """
        }

    def _classify_error(self, error_msg: str) -> ValidationErrorType:
        """Map error messages to error types"""
        error_patterns = {
            ValidationErrorType.STRING: ["string declaration", "string literal"],
            ValidationErrorType.SUBJECT: ["subject declaration", "missing field"],
            ValidationErrorType.ASSET: ["asset declaration", "invalid type", "invalid subtype"],
            ValidationErrorType.CLAUSE: ["clause declaration", "invalid consequence"],
            ValidationErrorType.TYPE: ["type mismatch", "invalid type"],
            ValidationErrorType.SYNTAX: ["unexpected token", "syntax error"]
        }
        
        error_msg = error_msg.lower()
        for error_type, patterns in error_patterns.items():
            if any(pattern.lower() in error_msg for pattern in patterns):
                return error_type
        return ValidationErrorType.SYNTAX

    def validate_and_generate(self, prompt: str) -> str:
        """Main entry point for Kelsen code generation with validation"""
        attempts = 0
        feedback_history = []
        current_prompt = prompt

        while attempts < self.max_attempts:
            try:
                # Generate code
                generated_code = self._generate_code(current_prompt)
                
                # Try parsing and validating
                parsed_output = self._parse_code(generated_code)
                validated_output = KelsenModelOutput.parse_obj(parsed_output)
                
                # Log successful generation
                logger.info(f"Successfully generated valid Kelsen code after {attempts + 1} attempts")
                return generated_code

            except pydantic.ValidationError as ve:
                error_type = self._classify_error(str(ve))
                feedback = ValidationFeedback(
                    is_valid=False,
                    error_type=error_type,
                    error_message=str(ve),
                    error_location=str([err["loc"] for err in ve.errors()]),
                    suggested_correction=self._get_relevant_examples([str(ve)])
                )
                feedback_history.append(feedback)
                
                # Create corrected prompt with examples
                current_prompt = self._create_correction_prompt(
                    generated_code, 
                    error_type
                )
                
            except Exception as e:
                logger.error(f"Unexpected error during generation: {str(e)}")
                raise

            attempts += 1

        raise ValueError(f"Failed to generate valid code after {self.max_attempts} attempts. Errors: {feedback_history}")

    def _get_relevant_examples(self, error_messages: List[str]) -> str:
            """Get relevant examples based on error messages"""
            examples = []
            
            # Look for keywords in error messages to determine which examples to show
            error_text = ' '.join(error_messages).lower()
            
            if 'string' in error_text:
                examples.append(('String Declaration', self.valid_examples['strings']))
            if 'asset' in error_text:
                examples.append(('Asset Declaration', self.valid_examples['assets']))
            if 'clause' in error_text:
                examples.append(('Clause Declaration', self.valid_examples['clauses']))
                
            # If no specific examples matched, return all examples
            if not examples:
                examples = [
                    ('String Declaration', self.valid_examples['strings']),
                    ('Asset Declaration', self.valid_examples['assets']),
                    ('Clause Declaration', self.valid_examples['clauses'])
                ]
                
            # Format examples
            formatted_examples = []
            for title, example in examples:
                formatted_examples.append(f"# {title}")
                formatted_examples.append(example.strip())
                
            return '\n\n'.join(formatted_examples)

class KelsenModel:
    def __init__(self, full_model_path: str, clauses_model_path: str, assets_model_path: str, device: str = "cuda"):
        """Initialize paths and tokenizer, but not models"""
        # Verify paths
        for path, name in [
            (full_model_path, "Full model"),
            (clauses_model_path, "Clauses model"),
            (assets_model_path, "Assets model")
        ]:
            if not os.path.exists(path):
                raise ValueError(f"{name} checkpoint path does not exist: {path}")
            
        self.device = torch.device(device)
        logger.info(f"Initializing with paths...")
        
        # Store paths
        self.model_paths = {
            'full': full_model_path,
            'clauses': clauses_model_path,
            'assets': assets_model_path
        }
        
        # Initialize tokenizer only
        _, self.tokenizer = self.load_model_and_tokenizer(full_model_path, device)
        
        # Initialize tracking
        self.current_model = None
        self.current_model_type = None
        logger.info("Initialization complete")
        
    def generate(self, prompt: str) -> str:
        """Generate complete Kelsen code using all models with Pydantic validation"""
        try:
            logger.info(f"Generating code for prompt: {prompt}")
            attempts = 0
            max_attempts = 3
            
            while attempts < max_attempts:
                try:
                    # Generate with all models
                    full_output = self._generate_with_model('full', prompt)
                    logger.debug(f"Full model output: {full_output}")
                    
                    assets_output = self._generate_with_model('assets', prompt)  # Fixed method name
                    logger.debug(f"Assets model output: {assets_output}")
                    
                    clauses_output = self._generate_with_model('clauses', prompt)
                    logger.debug(f"Clauses model output: {clauses_output}")
                    
                    # Merge outputs with proper method and variables
                    merged_output = self.merge_outputs(
                        full_output=full_output,
                        assets_output=assets_output,
                        clauses_output=clauses_output
                    )
                    logger.debug(f"Merged output: {merged_output}")
                    
                    # Validate with Pydantic
                    validated_output = KelsenModelOutput.parse_kelsen_code(merged_output)
                    
                    # Convert back to formatted code
                    final_output = validated_output.to_kelsen_code()
                    logger.info("Successfully generated and validated Kelsen code")
                    
                    return final_output
                    
                except pydantic.ValidationError as ve:
                    logger.warning(f"Validation error on attempt {attempts + 1}: {str(ve)}")
                    attempts += 1
                    if attempts >= max_attempts:
                        raise
                    
                except Exception as e:
                    logger.error(f"Generation error: {str(e)}")
                    raise
                    
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            logger.error("Error details:", exc_info=True)
            raise
    
    def _generate_with_model(self, model_type: str, prompt: str) -> str:        
        """Generate with a specific model"""
        try:
            model = self._load_model(model_type)
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_attention_mask=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=encoded.input_ids,
                    attention_mask=encoded.attention_mask,
                    max_length=512,
                    min_length=10,
                    num_beams=3,
                    temperature=0.2,
                    do_sample=True,
                    no_repeat_ngram_size=0,
                    repetition_penalty=1.0,
                    early_stopping=False,
                    use_cache=True
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)            
            return generated_text  # Remove validator post-processing as it's not defined

        except Exception as e:
            logger.error(f"Error generating with {model_type} model: {str(e)}")
            raise
            
        finally:
            torch.cuda.empty_cache()
            gc.collect()
    
    def merge_outputs(self, full_output: str, assets_output: str, clauses_output: str) -> str:
        """
        Merge outputs from all models into a single coherent output with validation
        
        Args:
            full_output: Output from full model
            assets_output: Output from assets model
            clauses_output: Output from clauses model
        """
        try:
            # Initialize temporary Pydantic model data
            temp_output = {
                'strings': [],
                'assets': [],
                'clauses': []
            }
            
            # Track identifiers to avoid duplicates
            string_ids = set()
            asset_ids = set()
            clause_ids = set()
            
            # Process strings from full model
            for decl in re.findall(r'string\s+\w+\s*=\s*"[^"]*"\s*;', full_output):
                match = re.match(r'string\s+(\w+)\s*=\s*"([^"]*)"', decl)
                if match and match.group(1) not in string_ids:
                    string_ids.add(match.group(1))
                    temp_output['strings'].append(
                        StringDeclaration(
                            identifier=match.group(1),
                            value=f'"{match.group(2)}"'
                        )
                    )
            
            # Process assets, preferring assets model output
            asset_declarations = re.findall(r'asset\s+\w+\s*=\s*[^;]+;', assets_output)
            if not asset_declarations:  # Fallback to full model
                asset_declarations = re.findall(r'asset\s+\w+\s*=\s*[^;]+;', full_output)
                
            for decl in asset_declarations:
                parts = re.match(r'asset\s+(\w+)\s*=\s*([^;]+)', decl)
                if parts and parts.group(1) not in asset_ids:
                    identifier = parts.group(1)
                    components = [p.strip() for p in parts.group(2).split(',')]
                    if len(components) >= 5:
                        asset_ids.add(identifier)
                        temp_output['assets'].append(
                            AssetDeclaration(
                                identifier=identifier,
                                asset_type=components[0],
                                subtype=components[1],
                                subject1=components[2],
                                action=components[3],
                                subject2=components[4]
                            )
                        )
            
            # Process clauses, preferring clauses model output
            clause_declarations = re.findall(r'clause\s+\w+\s*=\s*\{[^}]*\}\s*;', clauses_output)
            if not clause_declarations:  # Fallback to full model
                clause_declarations = re.findall(r'clause\s+\w+\s*=\s*\{[^}]*\}\s*;', full_output)
                
            for decl in clause_declarations:
                match = re.match(r'clause\s+(\w+)\s*=\s*\{([^}]*)\}', decl)
                if match and match.group(1) not in clause_ids:
                    identifier = match.group(1)
                    content = match.group(2)
                    try:
                        condition, consequence = content.split(',', 1)
                        
                        assets = [a.strip() for a in condition.split('AND')]
                        clause_ids.add(identifier)
                        temp_output['clauses'].append(
                            ClauseDeclaration(
                                identifier=identifier,
                                condition=Condition(
                                    assets=assets,
                                    operator='AND' if len(assets) > 1 else None
                                ),
                                consequence=self._parse_consequence(consequence.strip())
                            )
                        )
                    except ValueError as e:
                        logger.warning(f"Failed to parse clause {identifier}: {str(e)}")
                        continue
            
            # Create and validate final output
            validated = KelsenModelOutput(**temp_output)
            logger.info(f"Validated output with {len(temp_output['strings'])} strings, "
                    f"{len(temp_output['assets'])} assets, and {len(temp_output['clauses'])} clauses")
            
            # Return formatted code
            return validated.to_kelsen_code()
            
        except Exception as e:
            logger.error(f"Error merging outputs: {str(e)}")
            raise

    @staticmethod
    def _parse_consequence(consequence_str: str) -> LegalPosition:
        """Parse consequence string into LegalPosition"""
        match = re.match(r'(\w+)\((.*)\)', consequence_str)
        if not match:
            raise ValueError(f"Invalid consequence format: {consequence_str}")
            
        return LegalPosition(
            type=match.group(1),
            arg1=match.group(2)
        )

    def _load_model(self, model_type: str):
        """Load a specific model, unloading previous if needed"""
        if model_type not in self.model_paths:
            raise ValueError(f"Invalid model type: {model_type}. Must be one of: {list(self.model_paths.keys())}")
            
        if self.current_model_type == model_type and self.current_model is not None:
            return self.current_model
            
        # Clear current model
        if self.current_model is not None:
            del self.current_model
            torch.cuda.empty_cache()
            gc.collect()
        
        logger.info(f"Loading {model_type} model from {self.model_paths[model_type]}...")
        
        try:
            self.current_model, _ = self.load_model_and_tokenizer(
                self.model_paths[model_type], 
                str(self.device)
            )
            self.current_model_type = model_type
            
            if self.current_model is None:
                raise ValueError(f"Failed to load {model_type} model")
                
            logger.info(f"Successfully loaded {model_type} model")
            return self.current_model
            
        except Exception as e:
            logger.error(f"Error loading {model_type} model: {str(e)}")
            raise

    @staticmethod
    def load_model_and_tokenizer(model_path: str, device: str):
        peft_config = PeftConfig.from_pretrained(model_path)
        tokenizer = RobertaTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        
        special_tokens_dict = {'additional_special_tokens': list(extra_id_to_special_token.values())}
        tokenizer.add_special_tokens(special_tokens_dict)
        
        base_model = T5ForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path,
            load_in_8bit=True,
            device_map=None  # Disable device map
        )
        base_model.resize_token_embeddings(len(tokenizer))
        
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.to(device)  # Explicitly move model to the specified device
        
        log_memory_usage("loading model and tokenizer")
        return model, tokenizer

    def download_base_model():
        """
        Utility function to pre-download the base model
        """
        from transformers import T5ForConditionalGeneration, RobertaTokenizer
        
        try:
            base_model_name = "Salesforce/codet5-large"
            logger.info(f"Downloading base model {base_model_name}...")
            
            # Download and save the model
            T5ForConditionalGeneration.from_pretrained(base_model_name)
            RobertaTokenizer.from_pretrained(base_model_name)
            
            logger.info("Base model downloaded successfully")
        except Exception as e:
            logger.error(f"Error downloading base model: {str(e)}")
            raise
            
    def generate_batch(self, prompts: List[str], batch_size: int = 4) -> List[str]:
        """Generate Kelsen code for multiple prompts in batches"""
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            try:
                # Process batch in parallel
                with torch.no_grad():
                    inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                            return_tensors="pt").to(self.device)
                    outputs = self.model.generate(
                        **inputs,
                        max_length=512,
                        num_beams=3,
                        do_sample=True,
                        temperature=0.7
                    )
                    
                    decoded = [self.validator.post_process_output(
                        self.tokenizer.decode(out, skip_special_tokens=False)
                    ) for out in outputs]
                    results.extend(decoded)
                    
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size}: {str(e)}")
                results.extend([str(e)] * len(batch))
                
        return results


class KelsenCache:
    def __init__(self, maxsize=128):
        self.cache = lru_cache(maxsize=maxsize)(self._generate)
        
    def generate(self, prompt: str) -> str:
        """Generate valid Kelsen code from prompt"""
        try:
            logger.info(f"Generating code for prompt: {prompt}")
            
            # Properly tokenize with attention mask
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_attention_mask=True  # Explicitly request attention mask
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=encoded.input_ids,
                    attention_mask=encoded.attention_mask,  # Use the attention mask
                    max_length=512,
                    min_length=10,  # Reduced from 50 to match test_kelsen_model
                    num_beams=3,
                    do_sample=True,
                    temperature=0.2,  # Lower temperature for more focused sampling
                    no_repeat_ngram_size=0,
                    repetition_penalty=1.0,
                    early_stopping=False,
                    use_cache=True
                )
                
            # Decode the output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            logger.debug(f"Raw generated text: {generated_text}")
            
            # Post-process
            processed_text = self.validator.post_process_output(generated_text)
            logger.debug(f"Processed text: {processed_text}")
            
            return processed_text
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            logger.error("Error details:", exc_info=True)
            raise

def main():
    parser = argparse.ArgumentParser(description="Kelsen Code Generation")
    parser.add_argument("--full-model-path", required=True, help="Path to the full model checkpoint")
    parser.add_argument("--clauses-model-path", required=True, help="Path to the clauses model checkpoint")
    parser.add_argument("--assets-model-path", required=True, help="Path to the assets model checkpoint")
    parser.add_argument("--prompt", required=True, help="Input prompt for Kelsen code generation")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Set up logging based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    try:
        logger.info("Initializing model...")
        kelsen_model = KelsenModel(
            full_model_path=args.full_model_path,
            clauses_model_path=args.clauses_model_path,
            assets_model_path=args.assets_model_path,
            device=str(device)
        )
        
        logger.info("Starting generation...")
        generated_code = kelsen_model.generate(args.prompt)
        
        if generated_code:
            print("\nGenerated Kelsen Code:")
            print("="*50)
            print(generated_code)
            print("="*50)
        else:
            print("\nNo code was generated!")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error("Detailed error:", exc_info=True)
        raise

if __name__ == "__main__":
    main()