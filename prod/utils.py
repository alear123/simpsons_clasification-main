import torch
import torch.nn as nn
from torchvision.models import densenet121

CLASSES = [
    'abraham_grampa_simpson',
    'agnes_skinner',
    'apu_nahasapeemapetilon',
    'bart_simpson',
    'barney_gumble',
    'carl_carlson',
    'charles_montgomery_burns',
    'chief_wiggum',
    'cletus_spuckler',
    'comic_book_guy',
    'disco_stu',
    'edna_krabappel',
    'fat_tony',
    'gil',
    'groundskeeper_willie',
    'homer_simpson',
    'kent_brockman',
    'krusty_the_clown',
    'lenny_leonard',
    'lisa_simpson',
    'lionel_hutz',
    'maggie_simpson',
    'marge_simpson',
    'martin_prince',
    'mayor_quimby',
    'milhouse_van_houten',
    'miss_hoover',
    'moe_szyslak',
    'nelson_muntz',
    'ned_flanders',
    'otto_mann',
    'patty_bouvier',
    'principal_skinner',
    'professor_john_frink',
    'ralph_wiggum',
    'rainier_wolfcastle',
    'selma_bouvier',
    'sideshow_bob',
    'sideshow_mel',
    'snake_jailbird',
    'troy_mcclure',
    'waylon_smithers'
]

idx_to_class = {i: label for i, label in enumerate(CLASSES)}

class EmbeddingModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)

def load_model(path):
    try:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        base_model = densenet121(weights=None)
        base_model.classifier = nn.Linear(base_model.classifier.in_features, len(CLASSES))

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                base_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                base_model.load_state_dict(checkpoint['state_dict'])
            else:
                base_model.load_state_dict(checkpoint)
        else:
            base_model.load_state_dict(checkpoint)

        embedding_model = EmbeddingModel(base_model)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        base_model = densenet121(weights=None)
        base_model.classifier = nn.Linear(base_model.classifier.in_features, len(CLASSES))
        embedding_model = EmbeddingModel(base_model)

    return embedding_model, idx_to_class

def predict_character(model, img_tensor, idx_to_class, reference_embeddings=None):
    try:
        model.eval()
        with torch.no_grad():
            query_embedding = model(img_tensor)

            if reference_embeddings is None:
                embedding_norm = torch.norm(query_embedding, dim=1)
                predicted_idx = int(embedding_norm.item() * len(CLASSES)) % len(CLASSES)
                return idx_to_class.get(predicted_idx, "Desconocido"), query_embedding.cpu().numpy()

            similarities = torch.nn.functional.cosine_similarity(
                query_embedding,
                reference_embeddings,
                dim=1
            )

            predicted_idx = torch.argmax(similarities).item()
            confidence = similarities[predicted_idx].item()

            return idx_to_class.get(predicted_idx, "Desconocido"), confidence, query_embedding.cpu().numpy()

    except Exception as e:
        print(f"Error en la predicción: {e}")
        return "Error en la predicción", 0.0, None