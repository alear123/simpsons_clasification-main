import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from utils import load_model, predict_character
import traceback
import os

# Configuración de la página
st.set_page_config(
    page_title="Detector de Personajes de Los Simpsons",
    page_icon="🟡",
    layout="centered"
)

@st.cache_resource
def load_cached_model():
    """Carga el modelo y embeddings con manejo de errores mejorado"""
    try:
        model, idx_to_class = load_model('prod/modelo.pth')
        model.eval()

        # DIAGNÓSTICO: Verificar rutas y archivos
        st.write("🔍 **Diagnóstico de archivos:**")
        
        # Verificar directorio actual
        current_dir = os.getcwd()
        st.write(f"📁 Directorio actual: `{current_dir}`")
        
        # Verificar si existe la carpeta prod
        prod_dir = os.path.join(current_dir, 'prod')
        if os.path.exists(prod_dir):
            st.success("✅ Directorio 'prod' encontrado")
            
            # Listar archivos en prod/
            prod_files = os.listdir(prod_dir)
            st.write("📋 Archivos en prod/:")
            for file in prod_files:
                file_path = os.path.join(prod_dir, file)
                file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
                st.write(f"  • {file} ({file_size} bytes)")
        else:
            st.error("❌ Directorio 'prod' no encontrado")
            
            # Buscar archivos .pt en el directorio actual
            pt_files = [f for f in os.listdir('.') if f.endswith('.pt')]
            if pt_files:
                st.write("🔍 Archivos .pt encontrados en directorio actual:")
                for file in pt_files:
                    st.write(f"  • {file}")
        
        # Verificar archivo específico
        embeddings_path = 'prod/reference_embeddings.pt'
        if os.path.exists(embeddings_path):
            st.success(f"✅ Archivo {embeddings_path} encontrado")
            file_size = os.path.getsize(embeddings_path)
            st.write(f"📊 Tamaño: {file_size} bytes")
        else:
            st.error(f"❌ Archivo {embeddings_path} NO encontrado")
            
            # Buscar archivos similares
            possible_paths = [
                'reference_embeddings.pt',
                './prod/reference_embeddings.pt',
                os.path.join(os.getcwd(), 'prod', 'reference_embeddings.pt')
            ]
            
            st.write("🔍 Buscando en rutas alternativas:")
            for path in possible_paths:
                if os.path.exists(path):
                    st.success(f"✅ Encontrado en: {path}")
                    embeddings_path = path
                    break
                else:
                    st.write(f"❌ No encontrado: {path}")

        reference_embeddings = None
        try:
            # Intentar cargar con ruta absoluta
            abs_path = os.path.abspath(embeddings_path)
            st.write(f"🔄 Intentando cargar desde: `{abs_path}`")
            
            reference_embeddings = torch.load(embeddings_path, map_location='cpu')
            st.success("✅ Embeddings de referencia cargados correctamente")
            st.write(f"📐 Dimensiones: {reference_embeddings.shape}")
            st.write(f"📊 Tipo de datos: {reference_embeddings.dtype}")
            
        except FileNotFoundError as e:
            st.error(f"❌ Archivo no encontrado: {e}")
            st.write("💡 **Soluciones posibles:**")
            st.write("1. Verificar que el archivo existe en la ruta correcta")
            st.write("2. Verificar permisos de lectura del archivo")
            st.write("3. Regenerar el archivo reference_embeddings.pt")
            
        except Exception as e:
            st.error(f"❌ Error al cargar embeddings: {e}")
            st.code(traceback.format_exc())

        return model, idx_to_class, reference_embeddings, None
        
    except Exception as e:
        return None, None, None, f"{str(e)}\n{traceback.format_exc()}"

# Función alternativa para generar embeddings si no existen
def generate_reference_embeddings(model, idx_to_class):
    """Genera embeddings de referencia dummy (para pruebas)"""
    st.warning("🔧 Generando embeddings de referencia temporales...")
    
    # Crear embeddings dummy basados en las características del modelo
    with torch.no_grad():
        # Obtener dimensión de embedding del modelo
        dummy_input = torch.randn(1, 3, 128, 128)
        sample_embedding = model(dummy_input)
        embedding_dim = sample_embedding.shape[1]
        
        # Crear embeddings aleatorios para cada clase
        num_classes = len(idx_to_class)
        reference_embeddings = torch.randn(num_classes, embedding_dim)
        
        # Normalizar para mejorar similitud coseno
        reference_embeddings = torch.nn.functional.normalize(reference_embeddings, dim=1)
        
        st.info(f"📐 Embeddings temporales generados: {reference_embeddings.shape}")
        
        return reference_embeddings

model, idx_to_class, reference_embeddings, error_msg = load_cached_model()

# Si no hay embeddings pero sí modelo, generar temporales
if model is not None and reference_embeddings is None and error_msg is None:
    reference_embeddings = generate_reference_embeddings(model, idx_to_class)

# Transformación para imágenes
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Resto de la interfaz (igual que antes)
st.title("🟡 Detector de Personajes de Los Simpsons")
st.markdown("### Utilizando pérdida de las trillizas (Triplet Loss)")

if error_msg:
    st.error("❌ Error al cargar el modelo:")
    st.code(error_msg)
    st.stop()

st.info("📝 Sube una imagen de un personaje de Los Simpsons y el modelo intentará identificarlo.")

with st.expander("ℹ️ Personajes detectables"):
    cols = st.columns(3)
    for i, name in enumerate(idx_to_class.values()):
        with cols[i % 3]:
            st.write(f"• {name.replace('_', ' ').title()}")

uploaded_file = st.file_uploader(
    "📁 Subí una imagen",
    type=["jpg", "jpeg", "png"],
    help="Formatos soportados: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="📸 Imagen subida", use_column_width=True)

        with col2:
            with st.spinner("🔄 Analizando imagen..."):
                tensor = transform(image).unsqueeze(0)

                if reference_embeddings is not None:
                    prediction, confidence, query_embedding = predict_character(
                        model, tensor, idx_to_class, reference_embeddings
                    )
                    if prediction != "Error en la predicción":
                        st.success(f"🎯 **Personaje detectado:** {prediction.replace('_', ' ').title()}")
                        st.markdown(f"**Confianza:** {confidence:.3f}")
                    else:
                        st.error("❌ Error al procesar la imagen")
                else:
                    prediction, query_embedding = predict_character(
                        model, tensor, idx_to_class, None
                    )
                    if prediction != "Error en la predicción":
                        st.success(f"🎯 **Personaje detectado:** {prediction.replace('_', ' ').title()}")
                        st.warning("⚠️ Método alternativo sin embeddings de referencia")
                    else:
                        st.error("❌ Error al procesar la imagen")

    except Exception as e:
        st.error(f"❌ Error al procesar la imagen: {str(e)}")
        st.code(traceback.format_exc())

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Desarrollado con ❤️ - Redes Neuronales Profundas - UTN FRM
    </div>
    """, unsafe_allow_html=True
)