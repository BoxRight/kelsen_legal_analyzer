import torch
import logging
from typing import List, Dict, Any
from itertools import combinations
from collections import defaultdict
import sys
import numpy as np

class TruthTableAnalyzer:
    def __init__(self, matrix: torch.Tensor, atom_mapping: Dict[int, str]):
        """Initialize analyzer with matrix from Sesma product and atom mapping."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {self.device}")
        logging.info(f"Initializing TruthTableAnalyzer with matrix shape: {matrix.shape}")
        
        # Store atom mapping and variables
        self.atom_mapping = atom_mapping
        self.variables = [atom_mapping[i] for i in range(len(atom_mapping))]
        self.n_vars = len(self.variables)
        self.total_states = 2 ** self.n_vars
        
        # Convert matrix to GPU tensor properly
        self.matrix_tensor = matrix.clone().detach().to(device=self.device, dtype=torch.bool)
        self.valid_count = len(self.matrix_tensor)
        
        logging.info(f"Converting {self.valid_count} states to GPU tensor")
        self._compute_state_masks()
        
        logging.info(f"Found {self.valid_count} valid states out of {self.total_states} possible states")

    def _compute_state_masks(self):
        """Optimized version of state mask computation"""
        logging.info("Computing state masks...")
        BATCH_SIZE = 10000  # Process 10k valid states at a time
        
        # Initialize valid mask
        self.valid_mask = torch.zeros(self.total_states, dtype=torch.bool, device=self.device)
        
        # Convert valid states to integers for faster comparison
        logging.info("Converting states to integers...")
        valid_states_int = torch.zeros(len(self.matrix_tensor), dtype=torch.long, device=self.device)
        for i in range(self.n_vars):
            valid_states_int |= (self.matrix_tensor[:, i].long() << i)
        
        # Process in batches
        total_batches = (len(valid_states_int) + BATCH_SIZE - 1) // BATCH_SIZE
        logging.info(f"Processing {total_batches} batches...")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(valid_states_int))
            
            batch = valid_states_int[start_idx:end_idx]
            self.valid_mask[batch] = True
            
            if batch_idx % 10 == 0:
                logging.info(f"Processed batch {batch_idx + 1}/{total_batches}")
                torch.cuda.empty_cache()
        
        self.invalid_mask = ~self.valid_mask
        logging.info("State masks computation complete")

    def _calculate_state_entropy(self) -> float:
        """Calculate entropy using GPU tensors."""
        p_valid = self.valid_count / self.total_states
        if p_valid in [0, 1]:
            return 0.0
            
        p_valid_tensor = torch.tensor(p_valid, dtype=torch.float32, device=self.device)
        entropy = -(p_valid_tensor * torch.log2(p_valid_tensor) + 
                   (1 - p_valid_tensor) * torch.log2(1 - p_valid_tensor))
                   
        return float(entropy.cpu())

    def _calculate_information_gain(self, var_idx: int) -> float:
        """Calculate information gain for a variable."""
        var_col = self.matrix_tensor[:, var_idx]
        
        # Calculate entropy before split
        entropy_before = self._calculate_state_entropy()
        
        # Calculate entropy after split
        true_mask = var_col == True
        false_mask = ~true_mask
        
        p_true = float(true_mask.sum()) / len(var_col)
        p_false = 1 - p_true
        
        if p_true == 0 or p_false == 0:
            return 0.0
        
        true_states = self.matrix_tensor[true_mask]
        false_states = self.matrix_tensor[false_mask]
        
        entropy_true = float(-torch.sum(true_states.float().mean(dim=0) * 
                           torch.log2(true_states.float().mean(dim=0) + 1e-10)))
        entropy_false = float(-torch.sum(false_states.float().mean(dim=0) * 
                            torch.log2(false_states.float().mean(dim=0) + 1e-10)))
        
        entropy_after = p_true * entropy_true + p_false * entropy_false
        
        return entropy_before - entropy_after

    def get_state_distribution(self) -> Dict[str, float]:
        """Calculate the distribution of valid and invalid states."""
        logging.info("Calculating state distribution...")
        valid_count = int(self.valid_mask.sum().item())
        total_states = self.total_states
        invalid_count = total_states - valid_count
        
        distribution = {
            "total_possible_states": total_states,
            "valid_states": valid_count,
            "invalid_states": invalid_count,
            "valid_percentage": (valid_count / total_states) * 100,
            "entropy": self._calculate_state_entropy()
        }
        
        return distribution

    def analyze_critical_variables(self) -> Dict[str, Dict[str, float]]:
        """Analyze how each variable affects validity using GPU operations."""
        logging.info("Analyzing critical variables...")
        results = {}
        
        for i, var in enumerate(self.variables):
            var_col = self.matrix_tensor[:, i]
            
            true_mask = var_col == True
            false_mask = ~true_mask
            
            true_states = int(true_mask.sum().item())
            false_states = int(false_mask.sum().item())
            total_possible = 2 ** (self.n_vars - 1)
            
            info_gain = self._calculate_information_gain(i)
            
            results[var] = {
                "valid_when_true": true_states,
                "valid_when_false": false_states,
                "valid_true_percentage": float(true_states / total_possible * 100),
                "valid_false_percentage": float(false_states / total_possible * 100),
                "criticality_score": float(abs(true_states - false_states) / total_possible),
                "information_gain": float(info_gain)
            }
            
            if (i + 1) % 5 == 0:
                logging.info(f"Processed {i + 1}/{self.n_vars} variables")
        
        return results

    def analyze_conditional_probabilities(self) -> Dict[str, Dict[str, float]]:
        """Analyze probability of each variable given others using GPU operations."""
        logging.info("Analyzing conditional probabilities...")
        conditional_probs = defaultdict(dict)
        
        for i, var1 in enumerate(self.variables):
            var1_col = self.matrix_tensor[:, i]
            
            for value in [True, False]:
                mask = var1_col if value else ~var1_col
                if mask.any():
                    condition_states = self.matrix_tensor[mask]
                    probs = torch.mean(condition_states.float(), dim=0)
                    
                    for j, var2 in enumerate(self.variables):
                        if i != j:
                            key = f"{var2} when {var1}={value}"
                            conditional_probs[var1][key] = float(probs[j])
                    
                    marginal_prob = float(torch.mean(var1_col.float()))
                    conditional_probs[var1]["marginal_probability"] = marginal_prob
            
            if (i + 1) % 5 == 0:
                logging.info(f"Processed {i + 1}/{self.n_vars} variables")
        
        return dict(conditional_probs)

    def find_dependency_patterns(self) -> List[Dict[str, Any]]:
        """Identify patterns of variable dependencies using GPU operations."""
        logging.info("Finding dependency patterns...")
        dependencies = []
        
        var_pairs = list(combinations(range(self.n_vars), 2))
        total_pairs = len(var_pairs)
        
        for idx, (i, j) in enumerate(var_pairs):
            var1, var2 = self.variables[i], self.variables[j]
            
            var1_col = self.matrix_tensor[:, i]
            var2_col = self.matrix_tensor[:, j]
            
            true_mask = var1_col == True
            false_mask = ~true_mask
            
            if true_mask.any() and false_mask.any():
                p_true_given_true = float(torch.mean(var2_col[true_mask].float()))
                p_true_given_false = float(torch.mean(var2_col[false_mask].float()))
                
                if abs(p_true_given_true - p_true_given_false) > 0.1:
                    dependencies.append({
                        "if_variable": var1,
                        "then_variable": var2,
                        "strength": abs(p_true_given_true - p_true_given_false),
                        "p_true_given_true": p_true_given_true,
                        "p_true_given_false": p_true_given_false
                    })
            
            if (idx + 1) % 100 == 0:
                logging.info(f"Processed {idx + 1}/{total_pairs} variable pairs")
        
        return dependencies

    def find_mandatory_conditions(self) -> List[Dict[str, Any]]:
        """Find mandatory relationships between variables."""
        logging.info("Finding mandatory conditions...")
        conditions = []
        
        var_pairs = list(combinations(range(self.n_vars), 2))
        total_pairs = len(var_pairs)
        
        for idx, (i, j) in enumerate(var_pairs):
            var1, var2 = self.variables[i], self.variables[j]
            
            var1_col = self.matrix_tensor[:, i]
            var2_col = self.matrix_tensor[:, j]
            
            implies = torch.all((var1_col & ~var2_col) == False)
            reverse_implies = torch.all((var2_col & ~var1_col) == False)
            
            if implies:
                conditions.append({
                    "type": "implication",
                    "from_var": var1,
                    "to_var": var2,
                    "description": f"{var1.capitalize()} implica que {var2}"
                })
            if reverse_implies:
                conditions.append({
                    "type": "implication",
                    "from_var": var2,
                    "to_var": var1,
                    "description": f"{var2.capitalize()} implica que {var1}"
                })
            
            if (idx + 1) % 100 == 0:
                logging.info(f"Processed {idx + 1}/{total_pairs} variable pairs")
        
        return conditions

    def analyze_rule_conflicts(self) -> List[Dict[str, Any]]:
        """Identify potential conflicts between rules using GPU operations."""
        logging.info("Analyzing rule conflicts...")
        conflicts = []
        
        var_pairs = list(combinations(range(self.n_vars), 2))
        total_pairs = len(var_pairs)
        
        for idx, (i, j) in enumerate(var_pairs):
            var1, var2 = self.variables[i], self.variables[j]
            
            both_true = torch.any(self.matrix_tensor[:, i] & self.matrix_tensor[:, j])
            both_false = torch.any(~self.matrix_tensor[:, i] & ~self.matrix_tensor[:, j])
            
            if not both_true:
                conflicts.append({
                    "type": "mutual_exclusion",
                    "variables": (var1, var2),
                    "description": f"{var1} and {var2} cannot both be true"
                })
            elif not both_false:
                conflicts.append({
                    "type": "requirement",
                    "variables": (var1, var2),
                    "description": f"At least one of {var1} or {var2} must be true"
                })
            
            if (idx + 1) % 100 == 0:
                logging.info(f"Processed {idx + 1}/{total_pairs} variable pairs")
        
        return conflicts

    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """Perform all analyses with memory management."""
        try:
            logging.info("Starting comprehensive analysis")
            analysis = {}
        
            # Do analyses one at a time to manage memory
            analysis["state_distribution"] = self.get_state_distribution()
            torch.cuda.empty_cache()
        
            analysis["critical_variables"] = self.analyze_critical_variables()
            torch.cuda.empty_cache()
        
            analysis["mandatory_conditions"] = self.find_mandatory_conditions()
            torch.cuda.empty_cache()
        
            analysis["dependencies"] = self.find_dependency_patterns()
            torch.cuda.empty_cache()
        
            analysis["rule_conflicts"] = self.analyze_rule_conflicts()
            torch.cuda.empty_cache()
        
            logging.info("Comprehensive analysis complete")
            return analysis
        
        except torch.cuda.OutOfMemoryError:
            logging.error("GPU out of memory during analysis. Results may be incomplete.")
            return analysis  # Return partial results
        
    def __str__(self) -> str:
        """Generate a human-readable summary of the analysis."""
        analysis = self.get_comprehensive_analysis()
        
        summary = ["Truth Table Analysis Summary"]
        summary.append("=" * 30 + "\n")
        
        dist = analysis["state_distribution"]
        summary.append(f"Total States: {dist['total_possible_states']}")
        summary.append(f"Valid States: {dist['valid_states']} ({dist['valid_percentage']:.1f}%)")
        summary.append(f"Invalid States: {dist['invalid_states']}")
        summary.append(f"State Entropy: {dist['entropy']:.2f}\n")
        
        summary.append("Critical Variables:")
        for var, stats in analysis["critical_variables"].items():
            summary.append(f"{var}:")
            summary.append(f"  Valid when True: {stats['valid_when_true']} ({stats['valid_true_percentage']:.1f}%)")
            summary.append(f"  Valid when False: {stats['valid_when_false']} ({stats['valid_false_percentage']:.1f}%)")
            summary.append(f"  Criticality Score: {stats['criticality_score']:.2f}")
            summary.append(f"  Information Gain: {stats['information_gain']:.2f}\n")
        
        summary.append("Mandatory Conditions:")
        for cond in analysis["mandatory_conditions"]:
            summary.append(f"  {cond['description']}")
        
        summary.append("\nDependencies:")
        for dep in analysis["dependencies"]:
            summary.append(f"  {dep['if_variable']} -> {dep['then_variable']} (strength: {dep['strength']:.2f})")
        
        summary.append("\nRule Conflicts:")
        for conflict in analysis["rule_conflicts"]:
            summary.append(f"  {conflict['description']}")
        
        return "\n".join(summary)

def save_analysis_results(analyzer: TruthTableAnalyzer, output_file: str = "analysis_results.txt") -> bool:
    """Save analysis results to file with error handling."""
    try:
        logging.info(f"Saving analysis results to {output_file}...")
        analysis = analyzer.get_comprehensive_analysis()
        
        with open(output_file, "w") as f:
            f.write(str(analyzer))
            
        logging.info("Analysis results saved successfully")
        return True
        
    except torch.cuda.OutOfMemoryError:
        logging.error("GPU out of memory error. Try reducing batch size or freeing GPU memory.")
        return False
    except Exception as e:
        logging.error(f"Error saving analysis results: {str(e)}")
        return False

def process_truth_table():
    try:
        logging.info("Loading atom mapping...")
        atom_mapping = {}
        with open("atom_mapping.txt", "r") as f:
            for line in f:
                idx, atom = line.strip().split(" ", 1)
                atom_mapping[int(idx)] = atom
        
        logging.info("Loading matrix...")
        matrix_np = np.loadtxt("final_matrix.txt", dtype=np.int32)
        matrix = torch.from_numpy(matrix_np)
        analyzer = TruthTableAnalyzer(matrix, atom_mapping)
        # Save both technical analysis and legal report
        success = (
            save_analysis_results(analyzer, "technical_analysis.txt") and
            save_legal_report(analyzer, "informe_juridico.txt")
        )
        
        return success
        
    except Exception as e:
        logging.error(f"Error in analysis: {str(e)}")
        logging.error("Stack trace:", exc_info=True)
        return False
        
class LegalReportGenerator:
    def __init__(self, analyzer: TruthTableAnalyzer):
        self.analyzer = analyzer
        self.analysis = analyzer.get_comprehensive_analysis()
    
    def generate_legal_report(self, language='es') -> str:
        """Generate a formal legal report in Spanish or English."""
        if language == 'es':
            return self._generate_spanish_report()
        else:
            return self._generate_english_report()
    
    def _generate_spanish_report(self) -> str:
        """Generate Spanish legal report."""
        dist = self.analysis["state_distribution"]
        critical = self.analysis["critical_variables"]
        mandatory = self.analysis["mandatory_conditions"]
        dependencies = self.analysis["dependencies"]
        conflicts = self.analysis["rule_conflicts"]
        if dist['entropy'] == 0:
        	restrictive_level = f"Contrato muy restrictivo: {dist['entropy']:.2f}"
        elif dist['entropy'] > 0 and dist['entropy'] < .25:
        	restrictive_level =  f"Contrato restrictivo: {dist['entropy']:.2f}"
        elif dist['entropy'] > .25 and dist['entropy'] < .50:
        	restrictive_level = f"Contrato equilibrado: {dist['entropy']:.2f}"
        elif dist['entropy'] > .50 and dist['entropy'] < .75:
        	restrictive_level = f"Contrato permisivo: {dist['entropy']:.2f}"
        else:
        	restrictive_level = f"Contrato muy permisivo: {dist['entropy']:.2f}"
        
        report = [
            "DICTAMEN SOBRE ANÁLISIS PROBABILÍSTICO DE CONFIGURACIONES JURÍDICAS",
            "=" * 80 + "\n",
            
            "I. DISTRIBUCIÓN DE ESTADOS JURÍDICOS",
            "-" * 40,
            f"Total de configuraciones posibles: {dist['total_possible_states']:,}",
            f"Estados jurídicamente válidos: {dist['valid_states']:,} ({dist['valid_percentage']:.1f}%)",
            f"Estados jurídicamente inválidos: {dist['invalid_states']:,}",
            f"Nivel de restricción.- {restrictive_level}\n",
            
            "II. ANÁLISIS DE ELEMENTOS JURÍDICOS CRÍTICOS",
            "-" * 40
        ]
        
        # Sort critical variables by criticality score
        sorted_vars = sorted(
            critical.items(), 
            key=lambda x: x[1]['criticality_score'], 
            reverse=True
        )
        
        for var, stats in sorted_vars:
            if stats['criticality_score'] <= 0.29:
            	criticality_level = f"Elemento circunstancial: {stats['criticality_score']:.2f}"
            elif stats['criticality_score'] > 0.29 and stats['criticality_score'] < 0.59:
            	criticality_level = f"Elemento accidental: {stats['criticality_score']:.2f}"
            elif stats['criticality_score'] > 0.6 and stats['criticality_score'] < 0.89:
            	criticality_level = f"Elemento natural: {stats['criticality_score']:.2f}"
            else:
            	criticality_level = f"Elemento esencial: {stats['criticality_score']:.2f}"
            	
            if stats['information_gain'] <= -0.7:
            	information_level = f"Elemento antinómico: {stats['information_gain']:.2f}"
            elif stats['information_gain'] > -0.7 and stats['information_gain'] < -0.3:
            	information_level = f"Elemento disyuntivo: {stats['information_gain']:.2f}"
            elif stats['information_gain'] > -0.3 and stats['information_gain'] < 0.3:
            	information_level = f"Elemento supletorio: {stats['information_gain']:.2f}"
            elif stats['information_gain'] > 0.3 and stats['information_gain'] < 0.69:
                information_level = f"Elemento interpretativo: {stats['information_gain']:.2f}"
            else:
            	information_level = f"Elemento dispositivo: {stats['information_gain']:.2f}"
            
            report.extend([
                f"\nElemento: {var}",
                f"• Validez con elemento presente: {stats['valid_when_true']:,} ({stats['valid_true_percentage']:.1f}%)",
                f"• Validez con elemento ausente: {stats['valid_when_false']:,} ({stats['valid_false_percentage']:.1f}%)",
                f"• Índice de esencialidad.- {criticality_level}",
                f"• Índice de certeza obligacional.- {information_level}"
            ])
        
        report.extend([
            "\nIII. RELACIONES DE NECESIDAD JURÍDICA",
            "-" * 40
        ])
        
        for cond in mandatory:
            report.append(f"• {cond['description']}")
        
        report.extend([
            "\nIV. DEPENDENCIAS JURÍDICAS IDENTIFICADAS",
            "-" * 40,
            "\nA. Dependencias Fuertes (>0.40):"
        ])
        
        # Sort dependencies by strength
        strong_deps = [d for d in dependencies if d['strength'] > 0.40]
        weak_deps = [d for d in dependencies if 0.15 < d['strength'] <= 0.40]
        
        for dep in strong_deps:
            report.append(
                f"• {dep['if_variable']} → {dep['then_variable']} "
                f"(coeficiente: {dep['strength']:.2f})"
            )
        
        report.extend(["\nB. Dependencias Moderadas (0.15-0.40):"])
        
        for dep in weak_deps:
            report.append(
                f"• {dep['if_variable']} → {dep['then_variable']} "
                f"(coeficiente: {dep['strength']:.2f})"
            )
        
        report.extend([
            "\nV. INCOMPATIBILIDADES JURÍDICAS DETECTADAS",
            "-" * 40
        ])
        
        for conflict in conflicts:
            report.append(f"• {conflict['description']}")
        
        return "\n".join(report)

def save_legal_report(analyzer: TruthTableAnalyzer, output_file: str = "informe_juridico.txt"):
    """Generate and save legal report."""
    try:
        report_gen = LegalReportGenerator(analyzer)
        report = report_gen.generate_legal_report()
        
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(report)
        
        logging.info(f"Legal report saved to {output_file}")
        return True
        
    except Exception as e:
        logging.error(f"Error generating legal report: {str(e)}")
        return False
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = process_truth_table()
    sys.exit(0 if success else 1)
