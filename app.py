# app.py
from flask import Flask, render_template, request, jsonify, send_file
from urllib.parse import quote
import subprocess
import os
import sys
import logging
from typing import Optional
import shlex
import tempfile
import uuid
import json
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KelsenGenerator:
    def __init__(self):
        # Get absolute paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.full_model_path = os.path.abspath(os.path.join(base_dir, "results_Full/checkpoint-24720"))
        self.clauses_model_path = os.path.abspath(os.path.join(base_dir, "results_Clauses/checkpoint-10900"))
        self.assets_model_path = os.path.abspath(os.path.join(base_dir, "results_Assets/checkpoint-12560"))
        
        # Validate paths
        for path, name in [
            (self.full_model_path, "Full model"),
            (self.clauses_model_path, "Clauses model"),
            (self.assets_model_path, "Assets model")
        ]:
            if not os.path.exists(path):
                logger.error(f"{name} path does not exist: {path}")
                raise ValueError(f"{name} path does not exist: {path}")
            else:
                logger.info(f"Found {name} at: {path}")

    def generate_kelsen_code(self, input_text: str) -> Optional[str]:
        try:
            # Log the raw input
            logger.debug(f"Raw input text: {input_text}")
            
            # Clean input but preserve the full text
            cleaned_input = input_text.strip().replace('\n', ' ').replace('\r', ' ')
            # Properly escape the input for shell
            escaped_input = shlex.quote(cleaned_input)
            logger.debug(f"Cleaned input: {cleaned_input}")
            
            # Get absolute path to the script
            script_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                "pydantic_kelsen.py"
            ))
            
            # Build command with absolute paths
            cmd = [
                sys.executable,
                script_path,
                "--full-model-path", self.full_model_path,
                "--clauses-model-path", self.clauses_model_path,
                "--assets-model-path", self.assets_model_path,
                "--prompt", escaped_input,
                "--debug"
            ]
            
            logger.info(f"Executing command from {os.getcwd()}")
            logger.info(f"Command: {' '.join(cmd)}")
            
            # Set up environment
            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
            
            # Run the command
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                env=env,
                cwd=os.path.dirname(script_path)  # Set working directory
            )
            
            # Log complete output
            logger.debug(f"Process return code: {process.returncode}")
            logger.debug(f"Process stdout:\n{process.stdout}")
            logger.debug(f"Process stderr:\n{process.stderr}")
            
            if process.returncode != 0:
                logger.error(f"Command failed with return code {process.returncode}")
                return None
            
            # Parse output more carefully
            output_lines = process.stdout.split('\n')
            generated_code = []
            capture_output = False
            
            for line in output_lines:
                if "Generated Kelsen Code:" in line:
                    capture_output = True
                    continue
                elif "=" * 10 in line:  # More lenient boundary detection
                    if capture_output and generated_code:
                        break
                    continue
                elif capture_output and line.strip():
                    generated_code.append(line.strip())
            
            if not generated_code:
                logger.error("No generated code found in output")
                return None
                
            final_code = "\n".join(generated_code)
            logger.info(f"Successfully generated code:\n{final_code}")
            return final_code
            
        except Exception as e:
            logger.exception("Exception in generate_kelsen_code")
            return None

# Initialize generator
generator = KelsenGenerator()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No input data provided'
            })
        
        # Extract party information
        party1 = data.get('party1', {})
        party2 = data.get('party2', {})
        input_text = data.get('input_text', '').strip()
        
        # Validate input
        if not all([party1, party2, input_text]):
            return jsonify({
                'success': False,
                'error': 'Missing required information'
            })
            
        # Generate subject declarations
        subject_declarations = [
            f"subject {party1['role']} = \"{party1['name']}\", \"{party1['street']}\", {party1['phone']}, \"{party1['email']}\";",
            f"subject {party2['role']} = \"{party2['name']}\", \"{party2['street']}\", {party2['phone']}, \"{party2['email']}\";"
        ]
        
        # Generate Kelsen code
        generated_code = generator.generate_kelsen_code(input_text)
        if generated_code is None:
            return jsonify({
                'success': False,
                'error': 'Failed to generate Kelsen code'
            })
            
        # Insert subject declarations at the beginning of the code
        final_code = "\n".join(subject_declarations + [""] + [generated_code])
        
        # Generate unique ID for this code
        code_id = str(uuid.uuid4())
        
        return jsonify({
            'success': True,
            'code': final_code,
            'code_id': code_id,
            'download_url': f'/download/{code_id}?code={quote(final_code)}'
        })
        
    except Exception as e:
        logger.exception("Error in generate endpoint")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/download/<code_id>')
def download_file(code_id):
    try:
        # Get the generated code from session or database
        generated_code = request.args.get('code', '')
        if not generated_code:
            return jsonify({
                'success': False,
                'error': 'No code found to download'
            }), 404

        # Create a temporary file with .kelsen extension
        temp_dir = tempfile.gettempdir()
        filename = f"kelsen_code_{code_id}.kelsen"
        file_path = os.path.join(temp_dir, filename)

        # Write the code to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(generated_code)

        # Send the file
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='text/plain'
        )
    except Exception as e:
        logger.exception("Error creating download file")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        # Clean up temporary file
        try:
            os.remove(file_path)
        except:
            pass



@app.route('/execute/<code_id>', methods=['POST'])
def execute_kelsen(code_id):
    try:
        data = request.get_json()
        if not data or 'code' not in data:
            return jsonify({
                'success': False,
                'error': 'No code provided'
            }), 400

        generated_code = data['code']
        temp_dir = tempfile.gettempdir()
        kelsen_file = os.path.join(temp_dir, f"kelsen_code_{code_id}.kelsen")
        
        try:
            # Step 1: Write Kelsen code to file
            with open(kelsen_file, 'w', encoding='utf-8') as f:
                f.write(generated_code)

            # Step 2: Execute Kelsen
            kelsen_result = subprocess.run(
                ['./kelsen', kelsen_file],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Kelsen execution successful")

            # Step 3: Execute contract generator
            contract_result = subprocess.run(
                ['python', 'contract_generator.py'],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Read generated contract
            with open('generated_contract.txt', 'r', encoding='utf-8') as f:
                contract_text = f.read()

            # Step 4: Execute deontic logic analysis (now non-interactive)
            deontic_result = subprocess.run(
                ['./deonticLogic'],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Read legal report
            if os.path.exists('informe_juridico.txt'):
                with open('informe_juridico.txt', 'r', encoding='utf-8') as f:
                    legal_report = f.read()
            else:
                raise FileNotFoundError("Legal report file not generated")

            return jsonify({
                'success': True,
                'natural_language': contract_text,
                'legal_analysis': legal_report,
                'kelsen_output': kelsen_result.stdout
            })

        except subprocess.CalledProcessError as e:
            error_message = f"Process error: {e.stderr if e.stderr else str(e)}"
            logger.error(f"Error in execution pipeline: {error_message}")
            return jsonify({
                'success': False,
                'error': error_message
            })
        except FileNotFoundError as e:
            logger.error(f"File error: {str(e)}")
            return jsonify({
                'success': False,
                'error': f"File error: {str(e)}"
            })
        finally:
            # Cleanup temporary files
            for file in [kelsen_file, 'ast_output.json', 'generated_contract.txt', 'informe_jurídico.txt']:
                try:
                    if os.path.exists(file):
                        os.remove(file)
                        logger.debug(f"Cleaned up file: {file}")
                except Exception as e:
                    logger.error(f"Error cleaning up {file}: {str(e)}")

    except Exception as e:
        logger.exception("Error in execute endpoint")
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Initialize generator
generator = KelsenGenerator()

if __name__ == '__main__':
    app.run(debug=True)
