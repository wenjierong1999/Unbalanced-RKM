import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = './expr_results/expr_MNIST012_v2_1722637229/grouped_expr_results.csv'
data = pd.read_csv(file_path)

# Prepare the data for plotting
plot_data = data[['unbalance_ratio', 'model_name', 'mode_1_mean', 'mode_2_mean', 'mode_3_mean']]

# Define the grid dimensions (3 rows, 4 columns)
fig, axes = plt.subplots(3, 4, figsize=(15, 6), sharex=True, sharey=False)
#fig.suptitle('Mode Distributions by Model and Unbalance Ratio', fontsize=16)

# Define a color palette
#palette = sns.color_palette("Set2")
colors = ['blue', 'blue', 'red']


axis_label_fontsize = 17
tick_label_fontsize = 17
title_fontsize = 20

# Define model order to maintain consistency with previous steps
model_order = ['Vanilla RKM', 'RLS RKM (shared featuremap)', 'RLS RKM (pretrained classifier)', 'Iforestscore RKM']

model_name_map = {
    'Vanilla RKM': 'RKM',
    'RLS RKM (shared featuremap)': 'RLS-RKM (shared)',
    'RLS RKM (pretrained classifier)': 'RLS-RKM (class)',
    'Iforestscore RKM': 'Iforestscore RKM'

}

# Iterate over each subplot position with the updated layout
row = 0
col = 0
for unbalance_ratio in sorted(plot_data['unbalance_ratio'].unique()):
    for model_name in model_order:
        subset = plot_data[(plot_data['unbalance_ratio'] == unbalance_ratio) & (plot_data['model_name'] == model_name)]
        if not subset.empty:
            ax = axes[row, col]
            sns.barplot(x=['0', '1', '2'],
                        y=subset[['mode_1_mean', 'mode_2_mean', 'mode_3_mean']].values[0],
                        hue=['mode_1', 'mode_2', 'mode_3'],
                        palette=colors, ax=ax, dodge=False, legend=False)
            ax.set_title(f'{model_name_map[model_name]} - {unbalance_ratio}', fontsize = title_fontsize)
            ax.set_xlabel('',fontsize=axis_label_fontsize)
            ax.set_ylabel('',fontsize=axis_label_fontsize)
            ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
        col += 1
        if col == 4:
            col = 0
            row += 1

# Adjust layout and labels
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
for ax in axes.flat:
    ax.set(xlabel='', ylabel='')
# for ax in axes.flat:
#     for label in ax.get_xticklabels():
#         label.set_rotation(45)

# Show plot
plt.savefig('Outputs/fig/expr-mnist012-gen-dist.png', dpi=400)
plt.show()