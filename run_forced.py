import torch
import os
from portfolio_models import EnhancedPortfolioActor

# Carica il checkpoint
checkpoint = torch.load("results/portfolio/weights/checkpoint_ep299.pt", weights_only=False)

# Estrai i parametri del modello
actor_state_dict = checkpoint['actor_state_dict']

# Ottieni informazioni sulle dimensioni
for key, value in actor_state_dict.items():
    if 'weight' in key or 'bias' in key:
        print(f"{key}: {value.shape}")

# Crea un nuovo modello con le dimensioni corrette
state_size = 395  # Prendi questa dimensione dai log
action_size = 6
features_per_asset = 65  # Usa il numero corretto di feature
encoding_size = 32
attention_size = 34

# Crea un nuovo modello compatibile
new_model = EnhancedPortfolioActor(
    state_size=state_size,
    action_size=action_size,
    features_per_asset=features_per_asset,
    encoding_size=encoding_size,
    attention_size=attention_size
)

# Salva solo i pesi che corrispondono (solo per testing)
torch.save(new_model.state_dict(), "results/portfolio/weights/new_model.pth")

print("Modello di test creato")