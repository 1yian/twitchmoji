import dataset
import models.dan
import torch

dataset_file = 'dataset/hi.txt'
emote_file = 'dataset/emotes.txt'

epochs = 2000

train_dataset = dataset.TwitchEmoteDataset(dataset_file, emote_file)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=256, collate_fn= lambda x: x)
params = models.dan.DeepAveragingNetwork.get_default_params()
params['num_labels'] = len(train_dataset.emotes) + 1
model = models.dan.DeepAveragingNetwork(params,
                                        train_dataset.vocab_frequencies)
model.cuda()
for epoch in range(epochs):
    print(model.train_model(train_loader, weight=1. / (train_dataset.check_emote_counts() + 1)))
    predicted_indices = model.predict("wtf that was actually hype")[0]
    print([train_dataset.get_emote(idx) for idx in predicted_indices])
