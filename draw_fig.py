import matplotlib.pyplot as plt


fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 15), dpi=500)
# game = ['1-G1', '1-G2', '1-G3', '1-G4', '1-G5', '2-G1', '2-G2', '2-G3', '2-G4', '2-G5', '3-G1', '3-G2', '3-G3',
        # '3-G4', '3-G5', 'G1', '-G2', '-G3', '-G4', '-G5', '-G6']
rates = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
accs = {'gemma': {'ja': [58.12,57.97,54.06,49.84,44.06,41.09,36.56], 'en': [57.81,57.81,56.64,55.16,53.83,49.53,50.47], 'de': [50.42,50.36,48,47.15,46.15,44.09,44.29]},
        'llama3': {'ja': [26.25,16.09,17.5,46.56,46.56,59.53,56.09], 'en': [0.0,0.0,0.0,0.0,0.0,0.0,0.0], 'de': [0.0,0.0,0.0,0.0,0.0,0.0,0.0]}}
# accs = {'ja': {'gemma': [], 'llama3': []}}
colors = ['red', 'green', 'blue']
line_style = ['-', '--', '-.']
languages = ["aa", "bb", "cc"]

for i, (model_name, data) in enumerate(accs.items()):
    for j, (lang, y_data) in enumerate(data.items()):
        # import pdb; pdb.set_trace()
        axs[i,j].plot(rates, y_data, c='blue', linestyle='-', label='merged model')
        axs[i,j].axhline(y_data[0], c='orange', label='base model', linestyle='--')
        axs[i,j].axhline(y_data[-1], c='green', label='CP model', linestyle='--')
        axs[i,j].scatter(rates, y_data, c='blue')
        axs[i,j].legend(loc='best')
        axs[i,j].grid(True, linestyle='--', alpha=0.5)
        axs[i,j].set_xlabel("fuse weight of CP model", fontdict={'size': 16})
        axs[i,j].xaxis.set_ticks_position('bottom')
        axs[i,j].set_ylabel("Accurecy(%)", fontdict={'size': 16})
        axs[i,j].set_title(f"{model_name}_{lang}", fontdict={'size': 20})

        for a,b in zip(rates, y_data):
            axs[i,j].text(a, b, b, ha='center', va='bottom', fontsize=13)

# for i in range(3):
#     axs[i].plot(game, y_data[i], c=colors[i], linestyle=line_style[i])
#     axs[i].scatter(game, y_data[i], c=colors[i])
#     axs[i].legend(loc='best')
#     axs[i].set_yticks(range(0, 50, 5))
#     axs[i].grid(True, linestyle='--', alpha=0.5)
#     axs[i].set_xlabel("fuse weight of CP model", fontdict={'size': 16})
#     axs[i].set_ylabel("Accurecy(%)", fontdict={'size': 16}, rotation=0)
#     axs[i].set_title(f"{model_name}_{languages[i]}"), fontdict={'size': 20})
fig.autofmt_xdate()
# plt.show()
plt.savefig('img_results/results.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.1)