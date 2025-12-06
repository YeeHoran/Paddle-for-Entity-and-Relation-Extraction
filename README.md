# Paddle-for-Entity-and-Relation-Extraction

This is an initial try on constructing knowledge graph. The environment installations are listed as below, and model download and test code are under the code part.

1. Create and activate an environment.

2. Install Paddle
pip install paddlepaddle==2.6.2 paddlenlp==2.6.1 -i https://pypi.org/simple

result should like:
Successfully installed astor-0.8.1 decorator-5.2.1 geventhttpclient-2.0.2 paddlenlp-2.6.1 paddlepaddle-2.6.2 protobuf-3.20.2 tritonclient-2.51.0

3. Verify the environment and their consistency
python -c "import paddle; print('PaddlePaddle version:', paddle.__version__)"
python -c "import paddlenlp; print('PaddleNLP version:', paddlenlp.__version__)"
