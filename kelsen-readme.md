# Kelsen Web Interface

Web interface for generating and analyzing legal contracts using Kelsen code, with automated natural language generation and deontic logic analysis.

## Prerequisites

- Python 3.12+
- GHC (Glasgow Haskell Compiler)
- Flask
- PyTorch/Transformers library

## Installation

```bash
git clone [repository-url]
cd kelsen-web-interface

# Create and activate virtual environment
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Compile Haskell components
ghc -package containers -package process -package directory deonticLogic.hs
```

## Components

- `app.py`: Flask application
- `deonticLogic`: Legal analysis engine
- `contract_generator.py`: Natural language generator
- `pydantic_kelsen.py`: Kelsen code generator
- Checkpoints in:
  - `results_Full/checkpoint-24720`
  - `results_Clauses/checkpoint-10900`
  - `results_Assets/checkpoint-12560`

## Usage

1. Start server:
```bash
python app.py
```

2. Access interface at `http://localhost:5000`

3. Input:
   - Legal text
   - Party information
   - Execute analysis

## Outputs

- Kelsen code representation
- Natural language contract
- Legal analysis report
- Downloadable .kelsen files

## Files Generated

- `ast_output.json`: Abstract Syntax Tree
- `generated_contract.txt`: Natural language contract
- `informe_jurídico.txt`: Legal analysis report

## Architecture

```
Web Interface (Flask)
    ↓
Kelsen Code Generator
    ↓
DeonticLogic Analysis
    ↓
Contract Generator
```

## Error Handling

- Input validation
- File operation errors
- Execution pipeline errors
- Model errors

## Development

Built with:
- Flask (Backend)
- Tailwind CSS (Frontend)
- PyTorch (Models)
- Haskell (DeonticLogic)

## License

[License Information]

## Contact

[Contact Information]
