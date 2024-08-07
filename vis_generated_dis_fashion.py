import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = './expr_results/expr_Fashion_1722930660/grouped_expr_results.csv'
data = pd.read_csv(file_path)

print(data)

plot_data = data[['model_name', 'mode_1_mean', 'mode_2_mean', 'mode_3_mean',
                  'mode_4_mean', 'mode_5_mean', 'mode_6_mean', 'mode_7_mean', 'mode_8_mean', 'mode_9_mean', 'mode_10_mean']]

print(plot_data)

# Define the grid dimensions (3 rows, 4 columns)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)
#fig.suptitle('Mode Distributions by Model and Unbalance Ratio', fontsize=16)

# Define a color palette
#palette = sns.color_palette("Set2")

axis_label_fontsize = 17
tick_label_fontsize = 17
title_fontsize = 20

# Define model order to maintain consistency with previous steps
model_order = ['RKM', 'RLS-RKM (class)', 'RLS-RKM (shared)', 'Iforestscore RKM']
unbalanced_classes = [0,1,2,3,4,6,8]
colors = ['blue'] * 10
for i in unbalanced_classes:
    colors[i] = 'red'

print(colors)

row = 0
col = 0

for model_name in model_order:
    subset = plot_data[(plot_data['model_name'] == model_name)]
    if not subset.empty:
        ax = axes[row, col]
        sns.barplot(x=['0', '1', '2','3','4','5','6','7','8','9'],
                    y=subset[['mode_1_mean', 'mode_2_mean', 'mode_3_mean','mode_4_mean','mode_5_mean', 'mode_6_mean', 'mode_7_mean', 'mode_8_mean', 'mode_9_mean', 'mode_10_mean']].values[0],
                     ax=ax,palette=colors, dodge=False, legend=False)
        ax.set_title(f'{model_name}', fontsize = title_fontsize)
        ax.set_xlabel('',fontsize=axis_label_fontsize)
        ax.set_ylabel('',fontsize=axis_label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    col += 1
    if col == 2:
        col = 0
        row += 1
plt.savefig('Outputs/fig/expr-rls-fashion-gen-dist.png', dpi=600)
plt.show()