# Proyecto para detectar Personajes de Los Simpsons utilizando pérdida de las trillizas

Este proyecto implementa un sistema de reconocimiento de personajes de Los Simpsons mediante técnicas de aprendizaje profundo y aprendizaje de representaciones. Utiliza una red neuronal entrenada con pérdida de la trilliza (triplet loss), una estrategia que permite aprender un espacio de embedding donde imágenes del mismo personaje estén cercanas y de diferentes personajes estén alejadas.

El modelo fue entrenado sobre un conjunto de imágenes etiquetadas de los personajes más icónicos, y luego se utiliza para identificar nuevos rostros a partir de su representación vectorial.

## Clonar repositorio

    git clone https://github.com/crosaless/simpsons_clasification.git


## Instalar Dependencias

    pip install -r prod/requirements.txt

## Ejecutar aplicación

    streamlit run prod/app.py

O entra a la aplicación en Community Cloud:

(proximamente)