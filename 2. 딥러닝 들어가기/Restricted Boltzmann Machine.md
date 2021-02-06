> # Restricted Boltzmann Machine

- RBM, Unsupervised Learning
- AlexNet 직전인가, 두번째 부흥기의 시작점.
- 이해에 많은 도움을 주셨습니다. <https://velog.io/@sheep_jo/RBM-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EC%A0%95%EB%A6%AC>

---

Energy-based models

> <i>p(x) = e<sup>-E(x)</sup> / Z</i>

- 기댓값을 이용한 softmax 기법의 일종
- 이미지가 입력이라고 하시는 걸 보면 다차원 입력 가능인 듯.
- 그런 게 아니라 그냥 이런 구조체를 의미하는 거였음.
- 물리학적 관점에서 반비례 관계에 있는 이들을 Boltzmann distribution을 통해 표현할 수 있는데 이런 개념에서 시작했음.

1. General Boltzmann Machine
   - input끼리도 상호작용을 할 수 있고, hidden unit끼리도 상호작용을 할 수 있음.
   - 기본적으로 모든 유닛의 값을 0 또는 1의 binary 형태로 가정함.
2. Restricted Boltzmann Machine
   - GBM과 다르게 같은 계층의 unit과의 상호작용하지 않음.
   - 따라서 일부를 끊었다고 해서 restricted.
   - 일반적인 NN의 input & hidden layer와의 관계와 같은 형태
   - > 요약하면 RBM은 내부 상태 theta={W, a, b}의 값에 따라 에너지를 품고, 에너지는 샘플의 발생 확률을 규정한다. 따라서 어떤 샘플은 자주 발생하지만, 다른 샘플은 희소하게 발생하게 할 수 있다. (Stochastic 생성 모델)

<img src="images/RBM_element.JPG" style="display: block; margin: auto;" />

- <i>a<sub>i</sub> = W<sub>i</sub> * v<sub>i</sub> + b<sub>i</sub>, h<sub>i</sub> = activation(a<sub>i</sub>)</i>
- softmax 기법을 통해 확률을 구하고자 하면, 가능한 모든 수의 visible(input)과 hidden의 값을 모두 고려해야 하므로 한 번 계산하는데 시간이 많이 걸림.
- 그래서 Baysian's Rule을 통해 conditional distribution을 이용해 다르게 접근해고자 함.

```
깁스 샘플링(or MC sampling) 기법에 대해 찾아봐야 알겠지만, 개인적으로 깁스 샘플링을 통해 input을 샘플링(x)을 구성. (개념적으로 batch 구하기인 듯 함)

-> 이를 선형 결합과 sigmoid(activation function을 통과하듯)를 통과한 값보다 작은 임의의 수를 갖는 위치(1 ~ m)만 1을 주고 나머지는 0을 줌(h).

-> 이 hidden unit을 다시 선형 결합 및 sigmoid를 통과시켜 나온 값보다 작은 값에만 1을 주고 나머지는 0을 구하는 식으로 샘플링된 input(x')을 구함

-> 샘플링된 input을 통해 다시 hidden unit(h')을 구함

이후, 데이터에 더 의존적인 x, h(positive phase)와 덜 의존적인 x', h'의 차이를 learning rate으로 학습함. (대조 발산 알고리즘)
```

join distribution에 대한 monte carlo를 하듯 mc sampling을 진행함?

### Deep Belief Network (DBN)

- RBM을 쌓아서 계층을 구성. (until 2014)
- http://enginius.tistory.com/315
