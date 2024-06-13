
import torch

def angle_between( v1, v2):
    dot_prod = torch.sum(v1 * v2, dim=-1)
    norms = torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1)
    cos_theta = dot_prod / (norms + 1e-8)  # adding small epsilon for numerical stability
    angles = torch.acos(torch.clamp(cos_theta, -1, 1))  # clamp for numerical stability
    return angles

def circular_harmonics( theta, m):
    # e^(im*theta)
    real = torch.cos(m * theta)
    imag = torch.sin(m * theta)
    return real, imag

def calculate_torsion_embedded(delta_pos, torsion, L=10):
    # Calculate angles
    # batch_size, nodes_num, _, _ = delta_pos.size()
    # torsion_expanded = torsion.unsqueeze(2).expand(batch_size, nodes_num, nodes_num, 3)

    torsion_expand = torsion.unsqueeze(1).expand(-1, -1, delta_pos.size(1), -1)
    angles = angle_between(delta_pos, torsion_expand)
    
    # Calculate circular harmonics embeddings
    embeddings = []
    for m in range(-L, L+1):  # from -L to L
        real, imag = circular_harmonics(angles, m)
        embeddings.append(real.unsqueeze(-1))
        embeddings.append(imag.unsqueeze(-1))
    
    torsion_embedded = torch.cat(embeddings, dim=-1)
    
    return torsion_embedded


def angle_to_embedded(angles, L = 10):
    embeddings = []
    for m in range(-L, L+1):  # from -L to L
        real, imag = circular_harmonics(angles, m)
        embeddings.append(real.unsqueeze(-1))
        embeddings.append(imag.unsqueeze(-1))
    
    torsion_embedded = torch.cat(embeddings, dim=-1)
    return torsion_embedded

def angle_between( v1, v2):
    dot_prod = torch.sum(v1 * v2, dim=-1)
    norms = torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1)
    cos_theta = dot_prod / (norms + 1e-8)  # adding small epsilon for numerical stability
    angles = torch.acos(torch.clamp(cos_theta, -1, 1))  # clamp for numerical stability
    return angles

def angle_torsion(matrix1, matrix2, direction):
    # Calculate the dot product of two vectors
    # dot_product = (matrix1 * matrix2).sum(dim=-1)
    
    # Calculate the magnitude (norm) of each vector
    norm_matrix1 = torch.linalg.norm(matrix1, dim=-1)
    norm_matrix2 = torch.linalg.norm(matrix2, dim=-1)
    
    # Calculate the angle using arccosine
    # angles = torch.acos(torch.clamp(dot_product / (norm_matrix1 * norm_matrix2+1e-8), min=-1, max=1))
    
    # Calculate the cross product of the two vectors
    cross_product = torch.cross(matrix1, matrix2)
    
    # Check the direction of the cross product with respect to direction
    # sign = torch.sign((cross_product * direction).sum(dim=-1))
    
    # Adjust angles based on direction
    # torsions = torch.where(sign >= 0, angles, 2*torch.pi - angles)
    
    ## Another method , written on paper, to calculate the equivalent torsion angle:
    signed_sin_angles = ((cross_product/(norm_matrix1*norm_matrix2+1e-9).unsqueeze(-1))*direction).sum(dim=-1)
    torsions = torch.asin(torch.clamp(signed_sin_angles, min=-1, max=1))
    return torsions

# def angle_torsion(matrix1, matrix2, direction):
#     # Calculate the dot product of two vectors
#     dot_product = (matrix1 * matrix2).sum(dim=-1)
    
#     # Calculate the magnitude (norm) of each vector
#     norm_matrix1 = torch.linalg.norm(matrix1, dim=-1)
#     norm_matrix2 = torch.linalg.norm(matrix2, dim=-1)
    
#     # Calculate the angle using arccosine
#     angles = torch.acos(torch.clamp(dot_product / (norm_matrix1 * norm_matrix2+1e-8), min=-1, max=1))
    
#     # Calculate the cross product of the two vectors
#     cross_product = torch.cross(matrix1, matrix2)
    
#     # Check the direction of the cross product with respect to direction
#     sign = torch.sign((cross_product * direction).sum(dim=-1))
    
#     # Adjust angles based on direction
#     torsions = torch.where(sign >= 0, angles, 2*torch.pi - angles)
    
#     ## Another method , written on paper, to calculate the equivalent torsion angle:
#     # signed_sin_angles = ((cross_product/(norm_matrix1*norm_matrix2+1e-9).unsqueeze(-1))*direction).sum(dim=-1)
#     # torsions = torch.asin(torch.clamp(signed_sin_angles, min=-1, max=1))
#     return torsions

def e_angle_torsion(pos,edge_non_padding_mask):
    # 1. Calculate A_crossed
    #Normalize A
    A = (pos.unsqueeze(-3) - pos.unsqueeze(-2)).masked_fill(~edge_non_padding_mask.unsqueeze(-1), 0.0)
    A_norm = torch.linalg.norm(A, dim=-1, keepdim=True)
    A = A / (A_norm + 1e-8)
    ij_expanded = A.unsqueeze(-2).expand(-1, -1, -1, A.size(1), -1)  # shape: N x N x N x 3
    i_rows_expanded = A.unsqueeze(-3).expand(-1, -1, A.size(1), -1, -1)  # shape: N x N x N x 3
    j_column_expanded = A.unsqueeze(-4).expand(-1, A.size(1),-1, -1, -1) # shape: N x N x N x 3
    ij_i_crossed = torch.cross(ij_expanded, i_rows_expanded, dim=-1) # shape: N x N x N x 3
    ij_j_crossed = torch.cross(ij_expanded, j_column_expanded, dim=-1) # shape: N x N x N x 3
    ij_i_angle = angle_between(ij_expanded, i_rows_expanded) # shape: N x N x N
    ij_j_angle = angle_between(ij_expanded, j_column_expanded) # shape: N x N x N
    ij_i_angle = ij_i_angle.sum(dim=-1).masked_fill(~edge_non_padding_mask, 0.0)
    ij_j_angle = ij_j_angle.sum(dim=-1).masked_fill(~edge_non_padding_mask, 0.0)

    # 2. Calculate A_torsion
    ij_i_sum = ij_i_crossed.masked_fill(~edge_non_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0).sum(dim=-2) # shape: N x N x 3
    ij_j_sum = ij_j_crossed.masked_fill(~edge_non_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0).sum(dim=-2) # shape: N x N x 3

    torsions_matrix = angle_torsion(ij_i_sum, ij_j_sum, A).masked_fill(~edge_non_padding_mask, 0.0)

    return A, A_norm.squeeze(), ij_i_angle, ij_j_angle, torsions_matrix #Shape: N x N
