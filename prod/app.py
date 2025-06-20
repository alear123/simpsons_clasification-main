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

@st.cache_resource
def load_cached_model():
    """Carga el modelo y embeddings con manejo de errores"""
    try:
        model, idx_to_class = load_model('prod/modelo.pth')
        model.eval()

        reference_embeddings = None
        try:
            reference_embeddings = torch.load('prod/reference_embeddings.pt', map_location='cpu')
            st.info("✅ Embeddings de referencia cargados correctamente")
            st.write("📐 Dimensiones:", reference_embeddings.shape)
        except:
            st.warning("⚠️ No se encontraron embeddings de referencia. Usando método alternativo.")

        return model, idx_to_class, reference_embeddings, None
    except Exception as e:
        return None, None, None, f"{str(e)}\n{traceback.format_exc()}"

model, idx_to_class, reference_embeddings, error_msg = load_cached_model()

# Transformación para imágenes
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Interfaz de usuario
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