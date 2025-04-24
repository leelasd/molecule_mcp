#!/usr/bin/env python3
"""
Molecule Visualizer MCP Server

This server provides tools to visualize molecules and get their properties
using the RDKit library. It accepts SMILES strings and returns visualizations
and property information.
"""

import base64
import io
import logging
from typing import Dict, Optional

from mcp.server.fastmcp import FastMCP, Image
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, Lipinski
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image as PILImage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize the MCP server
mcp = FastMCP("Molecule Visualizer", dependencies=["rdkit", "pillow"])

# Dictionary of common molecules with their SMILES strings
COMMON_MOLECULES = {
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "glucose": "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O",
    "benzene": "C1=CC=CC=C1",
    "ethanol": "CCO",
    "water": "O",
    "methane": "C",
    "atp": "NC1=NC=NC2=C1N=CN2[C@@H]3[C@H]([C@H]([C@@H](O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O",
    "dopamine": "C1=CC(=C(C=C1CCN)O)O",
    "serotonin": "C1=CC2=C(C=C1O)C(=CN2)CCN",
    "cholesterol": "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C",
    "penicillin": "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C",
}


@mcp.tool()
def visualize_molecule(query: str, width: int = 400, height: int = 300, 
                      show_atom_indices: bool = False) -> Image:
    """Generate a 2D visualization of a molecule.
    
    Args:
        query: Molecule name or SMILES string
        width: Width of the output image in pixels
        height: Height of the output image in pixels
        show_atom_indices: Whether to show atom indices in the visualization
        
    Returns:
        Image object containing the molecule visualization
    """
    try:
        # Check if the query is a common molecule name
        if query.lower() in COMMON_MOLECULES:
            smiles = COMMON_MOLECULES[query.lower()]
            logger.info(f"Found common molecule: {query} -> {smiles}")
        else:
            smiles = query
        
        # Try to create a molecule from the SMILES string
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            raise ValueError(f"Could not parse '{query}' as a valid molecule. Please provide a valid SMILES string or use a common molecule name.")
        
        # Generate 2D coordinates if they don't exist
        if not mol.GetNumConformers():
            AllChem.Compute2DCoords(mol)
        
        # Create a drawing object
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        
        # Configure drawing options
        opts = drawer.drawOptions()
        if show_atom_indices:
            opts.addAtomIndices = True
        
        # Draw the molecule
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        # Get the PNG data
        png_data = drawer.GetDrawingText()
        
        # Return as Image object
        return Image(data=png_data, format="png")
    
    except Exception as e:
        logger.error(f"Error visualizing molecule: {str(e)}")
        raise


@mcp.tool()
def visualize_molecule_markdown(query: str, width: int = 400, height: int = 300, 
                               show_atom_indices: bool = False) -> str:
    """Generate a 2D visualization of a molecule and return as markdown.
    
    Args:
        query: Molecule name or SMILES string
        width: Width of the output image in pixels
        height: Height of the output image in pixels
        show_atom_indices: Whether to show atom indices in the visualization
        
    Returns:
        Markdown with embedded base64-encoded PNG image of the molecule
    """
    try:
        # Use the visualize_molecule tool to get the image
        image = visualize_molecule(query, width, height, show_atom_indices)
        
        # Convert to base64
        encoded_image = base64.b64encode(image.data).decode('utf-8')
        
        # Return the base64-encoded image with appropriate markdown for display
        return f"![Molecule: {query}](data:image/png;base64,{encoded_image})"
    
    except Exception as e:
        logger.error(f"Error visualizing molecule: {str(e)}")
        return f"Error visualizing molecule: {str(e)}"


@mcp.tool()
def get_molecule_properties(query: str) -> str:
    """Get properties of a molecule.
    
    Args:
        query: Molecule name or SMILES string
        
    Returns:
        Markdown formatted text with molecular properties
    """
    try:
        # Check if the query is a common molecule name
        if query.lower() in COMMON_MOLECULES:
            smiles = COMMON_MOLECULES[query.lower()]
            logger.info(f"Found common molecule: {query} -> {smiles}")
        else:
            smiles = query
        
        # Try to create a molecule from the SMILES string
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return f"Could not parse '{query}' as a valid molecule. Please provide a valid SMILES string or use a common molecule name."
        
        # Calculate basic properties
        mol_formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        mol_weight = Descriptors.ExactMolWt(mol)
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        num_rings = Chem.rdMolDescriptors.CalcNumRings(mol)
        
        # Calculate Lipinski properties
        h_donors = Lipinski.NumHDonors(mol)
        h_acceptors = Lipinski.NumHAcceptors(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        logp = Descriptors.MolLogP(mol)
        
        # Format the result
        result = f"# Molecule Properties: {query}\n\n"
        result += f"- **SMILES**: `{smiles}`\n"
        result += f"- **Molecular Formula**: {mol_formula}\n"
        result += f"- **Molecular Weight**: {mol_weight:.4f} g/mol\n"
        result += f"- **Number of Atoms**: {num_atoms}\n"
        result += f"- **Number of Bonds**: {num_bonds}\n"
        result += f"- **Number of Rings**: {num_rings}\n\n"
        
        result += "## Lipinski's Rule of Five Properties\n\n"
        result += f"- **Hydrogen Bond Donors**: {h_donors}\n"
        result += f"- **Hydrogen Bond Acceptors**: {h_acceptors}\n"
        result += f"- **Rotatable Bonds**: {rotatable_bonds}\n"
        result += f"- **LogP**: {logp:.4f}\n"
        
        return result
    
    except Exception as e:
        logger.error(f"Error getting molecule properties: {str(e)}")
        return f"Error getting molecule properties: {str(e)}"


@mcp.tool()
def get_common_molecules() -> str:
    """Get a list of common molecules that can be visualized.
    
    Returns:
        Markdown formatted list of common molecule names
    """
    molecules = list(COMMON_MOLECULES.keys())
    molecules.sort()
    
    result = "# Common Molecules\n\n"
    result += "These molecule names can be used directly with the `visualize_molecule` and `get_molecule_properties` tools:\n\n"
    
    for molecule in molecules:
        result += f"- {molecule}\n"
    
    return result


@mcp.resource("molecule://{name}/smiles")
def get_molecule_smiles(name: str) -> str:
    """Get the SMILES string for a common molecule.
    
    Args:
        name: Name of the common molecule
        
    Returns:
        SMILES string for the molecule
    """
    name_lower = name.lower()
    if name_lower in COMMON_MOLECULES:
        return COMMON_MOLECULES[name_lower]
    else:
        return f"Molecule '{name}' not found in common molecules database."


@mcp.resource("molecules://common")
def list_common_molecules() -> str:
    """List all common molecules available in the database.
    
    Returns:
        JSON formatted list of molecule names and their SMILES strings
    """
    import json
    return json.dumps(COMMON_MOLECULES, indent=2)


if __name__ == "__main__":
    # Run the server
    mcp.run()
