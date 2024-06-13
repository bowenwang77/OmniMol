import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
from PIL import Image
from rdkit.Geometry import Point2D
from scipy.spatial.distance import pdist, squareform
import os
def plot_mol_with_attn(atoms_full, smi, attn,save_dir, idx=0, with_hydrogen = True, prop_name = "molecule", pos = None):
    # Load molecule and compute 2D coordinates
    mol = Chem.MolFromSmiles(smi)
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    rdDepictor.Compute2DCoords(mol)

    def attention_to_color(attn_val):
        norm = Normalize(vmin=0, vmax=1)  # Adjusted normalization
        cmap = plt.cm.Reds
        return cmap(norm(attn_val)+0.2)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(20, 10))

    # Draw molecule and get transformed coordinates
    d = rdMolDraw2D.MolDraw2DCairo(400, 400)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    img_data = d.GetDrawingText()
    img = Image.open(BytesIO(img_data))
    ax.imshow(img)
    ax.axis('off')

    # Get transformed coordinates
    if with_hydrogen:
        non_hydrogen_indices = list(np.arange(atoms_full.shape[0]))
    else:
        non_hydrogen_indices = [index for index, a in enumerate(atoms_full) if a != 1]
    coords_2d = [Point2D(mol.GetConformer().GetAtomPosition(int(index)).x, mol.GetConformer().GetAtomPosition(int(index)).y) for index in non_hydrogen_indices]
    coords_transformed = [d.GetDrawCoords(coord) for coord in coords_2d]

    # Adjust the radius using a logarithmic scale
    def adjusted_radius(attn_val):
        base_radius = 14
        return base_radius + 30*attn_val
    hydrogen_lines = []
    # Calculate pairwise distances in the 3D space
    distances = squareform(pdist(pos))
    
    for index, atom_type in enumerate(atoms_full):
        if atom_type == 1:  # If it's a hydrogen atom
            # Exclude hydrogen atoms from the distances
            non_hydrogen_distances = distances[index, atoms_full != 1]
            
            # Get the index of the nearest non-hydrogen atom
            nearest_atom_idx = np.argmin(non_hydrogen_distances)
            
            # Add the attention value of the hydrogen atom to its nearest non-hydrogen atom
            attn[nearest_atom_idx] += attn[index]
            hydrogen_lines.append((index,nearest_atom_idx))

    if not with_hydrogen:
        # Filter out hydrogen atoms for plotting
        # atoms_full = atoms_full[atoms_full != 1]
        attn = attn[atoms_full != 1]
        pos = pos[atoms_full != 1]

    # Draw attention circles and display the 3D positions
    for coord, attn_val, position in zip(coords_transformed, attn, pos):
        color = attention_to_color(attn_val)
        circle = plt.Circle((coord.x, coord.y), adjusted_radius(attn_val), color=color, alpha=0.6)
        ax.add_patch(circle)

        # Display the 3D position as text next to the atom
        text_x = coord.x + 15  # Adjust these values as necessary to place the text correctly
        text_y = coord.y + 15
        position_text = f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), {attn_val:.2f}"
        ax.text(text_x, text_y, position_text, fontsize=5, ha='center', va='center')

    # Draw green lines for each pair in hydrogen_lines
    for hydrogen_idx, nearest_atom_idx in hydrogen_lines:
        start_coord = d.GetDrawCoords(Point2D(mol.GetConformer().GetAtomPosition(int(hydrogen_idx)).x, mol.GetConformer().GetAtomPosition(int(hydrogen_idx)).y))
        end_coord = d.GetDrawCoords(Point2D(mol.GetConformer().GetAtomPosition(int(nearest_atom_idx)).x, mol.GetConformer().GetAtomPosition(int(nearest_atom_idx)).y))
        ax.plot([start_coord.x, end_coord.x], [start_coord.y, end_coord.y], color="green", linewidth=10)

    # Colorbar
    norm = Normalize(vmin=0, vmax=1)  # Adjusted normalization
    cmap = plt.cm.Reds
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation="vertical")
    cbar.set_label("Attention")
    
    #Add title to my figure
    ax.set_title(prop_name+"___"+smi,fontsize=20)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if with_hydrogen:
        save_dir += "with_H/"
    else:
        save_dir += "no_H/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir+=prop_name+"_"+str(idx)
    plt.savefig(save_dir)  # Save with a transparent background
    plt.close()


if __name__ == '__main__':
    # Your data
    pos_full = np.array([[ 3.0552, -0.2574, -0.3570],
                        [ 2.3095,  0.3987,  0.3853],
                        [ 2.6988,  1.2024,  1.2423],
                        [ 0.8815,  0.2164,  0.2448],
                        [-0.3824,  1.0089,  1.1561],
                        [-1.5991,  0.1120,  0.2312],
                        [-2.8512,  0.2618,  0.4443],
                        [-0.9799, -0.6983, -0.6734],
                        [ 0.3927, -0.6417, -0.6671],
                        [-2.9618,  0.9513,  1.1974],
                        [-1.5081, -1.2926, -1.2974],
                        [ 0.9448, -1.2616, -1.3624]])
    atoms_full = np.array([8, 7, 8, 6, 16, 6, 7, 7, 6, 1, 1, 1])
    smi = 'O=[N+]([O-])C=1SC(=N)NC=1'
    attn = np.array([0.1, 0.1, 0.2, 0.01, 0.01, 0.3, 0.08, 0.02, 0.02, 0.02, 0.02, 0.02])
    save_dir = "admet_paper/visulization/mol_fig_toy/"
    plot_mol_with_attn(atoms_full, smi, attn,save_dir, idx=0, pos=pos_full, with_hydrogen = True)