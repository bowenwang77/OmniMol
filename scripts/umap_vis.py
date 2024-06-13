import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import umap

data = torch.load("./admet_paper/visualization_of_relation_meta/record/task_emb_final.pt")
label = torch.load("./admet_paper/visualization_of_relation_meta/record/task_idx_final.pt")

data = data[torch.logical_and(label!=1, label!=2)]
label = label[torch.logical_and(label!=1, label!=2)]


with open("./meta_dict3.json", 'r') as file:
    task_dict = json.load(file)

#DK for unkown type
prop_label={"Absorption":0,"Distribution":1,"Metabolism":2,"Excretion":3,"Toxicity":4,"Physiochemical":5,"DK":6}
prop_label_inverse={0:"Absorption",1:"Distribution",2:"Metabolism",3:"Excretion",4:"Toxicity",5:"Physiochemical",6:"DK"}
task_name_dict = {}
for task_name, task_info in list(task_dict.items())[:55]:
    task_idx = task_info['task_idx']
    task_isreg = task_info['regression']
    task_prop = task_info.get('prop', "DK")
    task_prop_label = prop_label[task_prop]
    task_name_dict[task_idx] = {"name": task_name,'regression':task_isreg, 'prop':task_prop,"prop_label":task_prop_label}


name_list = []
prop_label_list = []
is_reg_list = []
for i in label.tolist():
    name_list.append(task_name_dict[i]['name'])
    prop_label_list.append(task_name_dict[i]['prop_label'])
    is_reg_list.append(task_name_dict[i]['regression'])
is_reg_list = np.array(is_reg_list)

data_np = data[:,:].cpu().numpy()
label_np = np.array(prop_label_list)


# Apply UMAP
reducer = umap.UMAP(random_state=42)
embedded = reducer.fit_transform(data_np)

colormap = plt.cm.get_cmap('tab10', len(np.unique(label_np)))

reg_subset = embedded[is_reg_list]
not_reg_subset = embedded[~is_reg_list]

plt.figure(figsize=(12, 8))  # Increased figure size

plt.scatter(reg_subset[:, 0], reg_subset[:, 1], c=label_np[is_reg_list], cmap=colormap, marker='x', 
            label='Regression Tasks', s=300, vmin=label_np.min(), vmax=label_np.max(), alpha=0.7)  # Adjusted marker size and added alpha

for x, y, name in zip(reg_subset[:, 0], reg_subset[:, 1], np.array(name_list)[is_reg_list]):
    plt.text(x, y, r'${}$'.format(name), fontsize=14, ha='center')  # Increased font size

plt.scatter(not_reg_subset[:, 0], not_reg_subset[:, 1], c=label_np[~is_reg_list], cmap=colormap, marker='o', 
            label='Classification Tasks', s=300, vmin=label_np.min(), vmax=label_np.max(), alpha=0.7)  # Adjusted marker size and added alpha


for x, y, name in zip(not_reg_subset[:, 0], not_reg_subset[:, 1], np.array(name_list)[~is_reg_list]):
    plt.text(x, y, r'${}$'.format(name), fontsize=14, ha='center')  # Increased font size

for label in np.unique(label_np):
    plt.scatter([], [], color=colormap(label / (len(np.unique(label_np)) - 1)), label=prop_label_inverse[label])

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Moved the legend outside
plt.title('UMAP Visualization of Task Relationship', fontsize=14)  # Increased title font size
plt.xlabel('UMAP 1', fontsize=12)  # Increased axis label font size
plt.ylabel('UMAP 2', fontsize=12)  # Increased axis label font size
plt.xticks([])
plt.yticks([])
plt.grid(False)  # Enabled grid for better visual guidance

plt.savefig("./admet_paper/visualization_of_relation_meta/umap_vis_improved_no_ticks3", bbox_inches='tight')  # Adjusted to save with tight bounding box
