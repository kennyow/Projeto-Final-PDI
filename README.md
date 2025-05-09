# Jogo da Memória com Visão Computacional
Este projeto é um jogo da memória interativo que utiliza visão computacional em tempo real para identificar e parear imagens captadas por uma webcam, com base em um classificador treinado com o dataset CIFAR-10.

# Objetivo
Desenvolver um jogo da memória onde o usuário posiciona objetos ou imagens diante da webcam, e o sistema identifica a classe correspondente (por exemplo, "gato", "carro", "avião", etc.) utilizando um modelo treinado com CIFAR-10. Quando dois objetos da mesma classe forem detectados, o sistema reconhece o match e marca o par como encontrado.

# Tecnologias utilizadas
 - Python 3.x

 - OpenCV (para captura de vídeo)

 - TensorFlow / Keras ou PyTorch (para classificação com modelo treinado)

 - NumPy


🧠 Como funciona
O modelo é treinado previamente com o dataset CIFAR-10.

O jogo inicia a captura de vídeo pela webcam.

O usuário posiciona uma imagem ou objeto na frente da câmera.

O modelo classifica a imagem em tempo real.

O sistema registra a tentativa e verifica se houve um par correto.

Se o par for identificado (duas imagens da mesma classe), o match é confirmado.

▶️ Como rodar


Clone este repositório:

bash
Copiar
Editar
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
Instale as dependências:

bash
Copiar
Editar
pip install -r requirements.txt
Execute o jogo:

bash
Copiar
Editar
python main.py
📁 Estrutura do projeto
bash
Copiar
Editar
.
├── model/              # Modelo treinado com CIFAR-10
├── images/             # Imagens capturadas (opcional)
├── src/
│   ├── main.py         # Lógica principal do jogo
│   ├── camera.py       # Captura e pré-processamento da webcam
│   └── classifier.py   # Código do classificador
├── requirements.txt
└── README.md
🧪 Exemplos de classes reconhecidas
Avião

Automóvel

Gato

Cachorro

Sapo

Navio

Cavalo

📌 Possíveis melhorias
Interface gráfica com pontuação

Detecção de tempo de resposta

Aumento de classes com transfer learning

Reconhecimento de objetos reais com webcam

📜 Licença
Este projeto está licenciado sob a MIT License.

