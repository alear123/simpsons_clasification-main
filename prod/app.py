import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from utils import load_model, predict_character
import traceback

# Configuración de la página
st.set_page_config(
    page_title="Detector de Personajes de Los Simpsons",
    page_icon="🟡",
    layout="centered"
)

# Función para cargar el modelo con manejo de errores
@st.cache_resource
def load_cached_model():
    """Carga el modelo una sola vez y lo mantiene en caché"""
    try:
        model, idx_to_class = load_model('prod/modelo.pth')
        model.eval()
        
        # Intentar cargar embeddings de referencia si existen
        reference_embeddings = None
        try:
            reference_embeddings = torch.load('prod/reference_embeddings.pt', map_location='cpu')
            st.info("✅ Embeddings de referencia cargados correctamente")
        except:
            st.warning("⚠️ No se encontraron embeddings de referencia. Usando método alternativo.")
        
        return model, idx_to_class, reference_embeddings, None
    except Exception as e:
        error_msg = f"Error al cargar el modelo: {str(e)}\n{traceback.format_exc()}"
        return None, None, None, error_msg

# Cargar el modelo
model, idx_to_class, reference_embeddings, error_msg = load_cached_model()

# Transformación de imágenes
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Interfaz de usuario
st.title("🟡 Detector de Personajes de Los Simpsons")
st.markdown("### Utilizando pérdida de las trillizas (Triplet Loss)")

# Mostrar error si el modelo no se pudo cargar
if error_msg:
    st.error("❌ Error al cargar el modelo:")
    st.code(error_msg)
    st.stop()

# Información sobre el modelo
st.info("📝 Sube una imagen de un personaje de Los Simpsons y el modelo intentará identificarlo.")

# Expandir con información adicional
with st.expander("ℹ️ Información del modelo"):
    st.write("**Personajes detectables:**")
    if idx_to_class:
        # Mostrar personajes en columnas
        cols = st.columns(3)
        for i, character in enumerate(idx_to_class.values()):
            with cols[i % 3]:
                st.write(f"• {character.replace('_', ' ').title()}")

# Subida de archivo
uploaded_file = st.file_uploader(
    "📁 Subí una imagen", 
    type=["jpg", "jpeg", "png"],
    help="Formatos soportados: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    try:
        # Mostrar imagen subida
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="📸 Imagen subida", use_column_width=True)
        
        with col2:
            # Procesar imagen
            with st.spinner("🔄 Analizando imagen..."):
                tensor = transform(image).unsqueeze(0)
                
                # Predecir usando embeddings
                if reference_embeddings is not None:
                    prediction, confidence, query_embedding = predict_character(
                        model, tensor, idx_to_class, reference_embeddings
                    )
                    
                    # Mostrar resultado con confianza
                    if prediction != "Error en la predicción":
                        st.success(f"🎯 **Personaje detectado:**")
                        st.markdown(f"### {prediction.replace('_', ' ').title()}")
                        st.markdown(f"**Confianza:** {confidence:.3f}")
                        
                        # Mostrar embedding info
                        if query_embedding is not None:
                            st.markdown(f"**Dimensión del embedding:** {query_embedding.shape}")
                    else:
                        st.error("❌ Error al procesar la imagen")
                else:
                    # Método alternativo sin embeddings de referencia
                    prediction, query_embedding = predict_character(
                        model, tensor, idx_to_class, None
                    )
                    
                    if prediction != "Error en la predicción":
                        st.success(f"🎯 **Personaje detectado:**")
                        st.markdown(f"### {prediction.replace('_', ' ').title()}")
                        st.warning("⚠️ Predicción basada en método alternativo (sin embeddings de referencia)")
                        
                        if query_embedding is not None:
                            st.markdown(f"**Dimensión del embedding:** {query_embedding.shape}")
                    else:
                        st.error("❌ Error al procesar la imagen")
        
        # Información adicional
        st.markdown("---")
        st.info("💡 **Tip:** Para mejores resultados, usa imágenes claras con el personaje bien visible.")
        
    except Exception as e:
        st.error(f"❌ Error al procesar la imagen: {str(e)}")
        st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Desarrollado con ❤️ para el Trabajo Práctico Integrador<br>
        Redes Neuronales Profundas - Ingeniería en Sistemas de Información
    </div>
    """, 
    unsafe_allow_html=True
)
