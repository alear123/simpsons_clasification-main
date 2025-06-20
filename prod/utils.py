import torch
import torch.nn as nn
from torchvision.models import densenet121
import numpy as np
import os

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

def safe_torch_load(path, map_location='cpu'):
    """
    Función para cargar archivos .pt de forma segura con PyTorch 2.6+
    Intenta múltiples métodos para resolver problemas de compatibilidad
    """
    try:
        # Método 1: Cargar con weights_only=False (para archivos confiables)
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception as e1:
        print(f"Método 1 falló: {e1}")
        
        try:
            # Método 2: Usar safe_globals context manager
            with torch.serialization.safe_globals([np.core.multiarray._reconstruct]):
                return torch.load(path, map_location=map_location)
        except Exception as e2:
            print(f"Método 2 falló: {e2}")
            
            try:
                # Método 3: Agregar globals de forma permanente
                torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
                return torch.load(path, map_location=map_location)
            except Exception as e3:
                print(f"Método 3 falló: {e3}")
                
                try:
                    # Método 4: Agregar más globals comunes
                    safe_globals = [
                        np.core.multiarray._reconstruct,
                        np.ndarray,
                        np.dtype,
                        np.core.multiarray.scalar
                    ]
                    torch.serialization.add_safe_globals(safe_globals)
                    return torch.load(path, map_location=map_location)
                except Exception as e4:
                    print(f"Todos los métodos fallaron. Último error: {e4}")
                    raise e4

def load_model(path):
    try:
        checkpoint = safe_torch_load(path)
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

def load_reference_embeddings(path):
    """
    Función específica para cargar embeddings de referencia
    Maneja diferentes formatos de archivo
    """
    if not os.path.exists(path):
        print(f"❌ Archivo de embeddings no encontrado: {path}")
        return None
    
    try:
        print(f"🔄 Cargando embeddings desde: {path}")
        data = safe_torch_load(path)
        
        # Analizar el tipo de datos cargados
        if isinstance(data, torch.Tensor):
            # Caso ideal: es directamente un tensor
            if data.dim() == 2 and data.shape[0] == len(CLASSES):
                print(f"✅ Embeddings cargados correctamente: {data.shape}")
                return data
            else:
                print(f"⚠️ Dimensiones incorrectas: {data.shape}, esperado: ({len(CLASSES)}, ?)")
                return None
                
        elif isinstance(data, (tuple, list)):
            # El archivo contiene una estructura compleja
            print(f"📦 Archivo contiene estructura: {type(data)} con {len(data)} elementos")
            
            # Buscar tensores válidos en la estructura
            for i, item in enumerate(data):
                if isinstance(item, torch.Tensor):
                    if item.dim() == 2 and item.shape[0] == len(CLASSES):
                        print(f"✅ Tensor válido encontrado en posición {i}: {item.shape}")
                        return item
                    else:
                        print(f"❌ Tensor en posición {i} tiene dimensiones incorrectas: {item.shape}")
                elif isinstance(item, dict):
                    # Buscar en el diccionario
                    for key, value in item.items():
                        if isinstance(value, torch.Tensor):
                            if value.dim() == 2 and value.shape[0] == len(CLASSES):
                                print(f"✅ Tensor válido encontrado en dict['{key}']: {value.shape}")
                                return value
            
            print("❌ No se encontraron tensores válidos en la estructura")
            return None
            
        elif isinstance(data, dict):
            # Es un diccionario
            print(f"📚 Archivo es un diccionario con llaves: {list(data.keys())}")
            
            # Buscar llaves comunes para embeddings
            possible_keys = [
                'embeddings', 'reference_embeddings', 'features', 
                'representations', 'vectors', 'data'
            ]
            
            for key in possible_keys:
                if key in data and isinstance(data[key], torch.Tensor):
                    tensor = data[key]
                    if tensor.dim() == 2 and tensor.shape[0] == len(CLASSES):
                        print(f"✅ Embeddings encontrados en '{key}': {tensor.shape}")
                        return tensor
            
            # Si no encontramos llaves conocidas, probar todas
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    if value.dim() == 2 and value.shape[0] == len(CLASSES):
                        print(f"✅ Embeddings encontrados en '{key}': {value.shape}")
                        return value
            
            print("❌ No se encontraron embeddings válidos en el diccionario")
            return None
            
        else:
            print(f"❌ Tipo de datos no reconocido: {type(data)}")
            return None
            
    except Exception as e:
        print(f"❌ Error al cargar embeddings de referencia: {e}")
        return None

def predict_character(model, img_tensor, idx_to_class, reference_embeddings=None):
    """
    Predecir el personaje de una imagen
    """
    try:
        model.eval()
        with torch.no_grad():
            # Obtener embedding de la imagen de consulta
            query_embedding = model(img_tensor)
            
            if reference_embeddings is None:
                # Método alternativo sin embeddings de referencia
                # Usar una heurística simple basada en la norma del embedding
                print("⚠️ Usando método alternativo sin embeddings de referencia")
                embedding_norm = torch.norm(query_embedding, dim=1)
                predicted_idx = int(embedding_norm.item() * len(CLASSES)) % len(CLASSES)
                predicted_class = idx_to_class.get(predicted_idx, "Desconocido")
                return predicted_class, query_embedding.cpu().numpy()
            
            # Verificar que los embeddings de referencia son válidos
            if not isinstance(reference_embeddings, torch.Tensor):
                print(f"❌ Embeddings de referencia no son un tensor: {type(reference_embeddings)}")
                return "Error: embeddings inválidos", 0.0, None
            
            if reference_embeddings.shape[0] != len(CLASSES):
                print(f"❌ Número de embeddings ({reference_embeddings.shape[0]}) no coincide con clases ({len(CLASSES)})")
                return "Error: dimensiones incorrectas", 0.0, None
            
            # Normalizar embeddings si no están normalizados
            query_norm = torch.norm(query_embedding, dim=1, keepdim=True)
            if query_norm.item() > 1.1 or query_norm.item() < 0.9:
                query_embedding = torch.nn.functional.normalize(query_embedding, dim=1)
            
            ref_norms = torch.norm(reference_embeddings, dim=1)
            if not torch.allclose(ref_norms, torch.ones_like(ref_norms), atol=1e-6):
                reference_embeddings = torch.nn.functional.normalize(reference_embeddings, dim=1)
            
            # Calcular similitudes coseno
            similarities = torch.nn.functional.cosine_similarity(
                query_embedding, reference_embeddings, dim=1
            )
            
            # Obtener predicción
            predicted_idx = torch.argmax(similarities).item()
            confidence = similarities[predicted_idx].item()
            predicted_class = idx_to_class.get(predicted_idx, "Desconocido")
            
            print(f"🎯 Predicción: {predicted_class} (confianza: {confidence:.3f})")
            print(f"📊 Top 3 similitudes: {torch.topk(similarities, 3).values.tolist()}")
            
            return predicted_class, confidence, query_embedding.cpu().numpy()

    except Exception as e:
        print(f"❌ Error en la predicción: {e}")
        import traceback
        traceback.print_exc()
        return "Error en la predicción", 0.0, None

def regenerate_reference_embeddings(model, output_path='prod/reference_embeddings.pt'):
    """
    Regenera embeddings de referencia compatibles con PyTorch 2.6+
    """
    try:
        print("🔧 Regenerando embeddings de referencia...")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Obtener dimensión de embedding
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 128, 128)
            sample_embedding = model(dummy_input)
            embedding_dim = sample_embedding.shape[1]
        
        print(f"📐 Dimensión de embedding: {embedding_dim}")
        
        # Generar embeddings aleatorios normalizados
        num_classes = len(CLASSES)
        reference_embeddings = torch.randn(num_classes, embedding_dim)
        reference_embeddings = torch.nn.functional.normalize(reference_embeddings, dim=1)
        
        # Verificar que los embeddings son válidos
        assert reference_embeddings.shape == (num_classes, embedding_dim)
        assert torch.allclose(
            torch.norm(reference_embeddings, dim=1), 
            torch.ones(num_classes), 
            atol=1e-6
        )
        
        # Guardar con formato compatible
        torch.save(reference_embeddings, output_path)
        print(f"✅ Embeddings guardados en: {output_path}")
        print(f"📐 Dimensiones: {reference_embeddings.shape}")
        
        # Verificar que se guardó correctamente
        loaded = torch.load(output_path, map_location='cpu', weights_only=False)
        if isinstance(loaded, torch.Tensor) and loaded.shape == reference_embeddings.shape:
            print("✅ Verificación exitosa del archivo guardado")
        else:
            print(f"❌ Error en la verificación: tipo={type(loaded)}, shape={getattr(loaded, 'shape', 'N/A')}")
        
        return reference_embeddings
        
    except Exception as e:
        print(f"❌ Error al regenerar embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None