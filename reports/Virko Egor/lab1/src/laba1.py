import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size=0, learning_rate=0.1):
        self.X = np.array([])
        self.w = np.random.uniform(-1.0, 1.0, input_size + 1)  # w[0] — bias/threshold
        self.learning_rate = learning_rate
        self.target = np.array([])

    def set_X(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            print("X должен быть 1D или 2D массивом")
            return
        self.X = np.insert(X, 0, -1, axis=1)  # добавляем столбец -1 для bias

    def set_target(self, target):
        if len(target) != len(self.X):
            print(f"Длина target должна быть {len(self.X)}, получено {len(target)}")
            return
        self.target = np.array(target)

    def activate(self, arr_wsum):
        return 2 / (1 + np.exp(-arr_wsum)) - 1   # tanh-подобная (-1..+1)

    def der_activate(self, y_pred):
        return 0.5 * (1 - y_pred**2)

    def forward(self, X_input=None):
        X = self.X if X_input is None else X_input
        if X.size == 0:
            print("Вход X не задан!")
            return None
        wsum = np.dot(X, self.w)
        return self.activate(wsum)

    def train(self, epochs=1000):
        mse_history = []
        for _ in range(epochs):
            y_pred = self.forward()
            error = y_pred - self.target
            mse = np.mean(error**2)
            mse_history.append(mse)
            # Градиентный спуск (batch)
            grad = self.learning_rate * error * self.der_activate(y_pred)
            self.w -= np.dot(grad, self.X)
        return mse_history


# ── ТВОЙ ВАРИАНТ ────────────────────────────────────────────────
X_train = np.array([
    [ 2,  4],
    [-2,  4],
    [ 2, -4],
    [-2, -4]
])

Y_targets = np.array([0, 0, 1, 1])   # как в таблице (e)

# ────────────────────────────────────────────────────────────────

plt.figure(figsize=(12, 5))

# График сходимости для разных learning rate
plt.subplot(1, 2, 2)
for lr in [0.01, 0.1, 0.5]:
    p_test = Perceptron(input_size=2, learning_rate=lr)
    p_test.set_X(X_train)
    p_test.set_target(Y_targets)
    history = p_test.train(epochs=800)
    plt.plot(history, label=f'lr = {lr}')

plt.title("MSE в зависимости от эпох")
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)

# Основной перцептрон, который будем использовать
p = Perceptron(input_size=2, learning_rate=0.1)
p.set_X(X_train)
p.set_target(Y_targets)
p.train(epochs=1500)  # обычно хватает 1000–2000

def plot_current_state(user_point=None, user_class=None):
    plt.subplot(1, 2, 1)
    plt.cla()

    # Сетка для отрисовки разделяющей поверхности
    xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
    grid = np.c_[np.full(xx.ravel().shape, -1), xx.ravel(), yy.ravel()]  # bias -1
    Z = p.forward(grid).reshape(xx.shape)

    # Раскраска областей
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], colors=["#d62728", "white", "#1f77b4"], alpha=0.4)
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=1.5, linestyles='--')

    # Точки обучения
    colors = ['red' if y == 0 else 'blue' for y in Y_targets]
    plt.scatter(X_train[:,0], X_train[:,1], c=colors, s=120, edgecolors='black', linewidth=1.2, label='Обучающая выборка')

    if user_point is not None:
        pred = user_class
        col = 'yellow' if abs(pred) < 0.3 else ('lime' if pred > 0 else 'orange')
        plt.scatter(user_point[0], user_point[1], c=col, marker='*', s=320, edgecolors='black', zorder=10,
                    label=f'Тест: {pred:.3f}')

    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.legend(loc='upper right')
    plt.title("Разделяющая поверхность")
    plt.pause(0.08)


plot_current_state()
plt.show(block=False)

print("Вводи координаты точки (x1 x2)  или 'exit' для выхода")
print("Диапазон желательно [-5, 5], но можно любой")

try:
    while True:
        line = input("→ ").strip()
        if line.lower() in ['exit', 'выход', 'q']:
            break

        try:
            x1, x2 = map(float, line.split())
            user_input = np.array([[-1, x1, x2]])  # bias -1
            pred = p.forward(user_input)[0]

            cls = 1 if pred > 0 else 0
            print(f"Выход: {pred:>6.4f}  →  класс {cls}  (0 — красный, 1 — синий)")

            plot_current_state([x1, x2], pred)
            plt.draw()
        except:
            print("Ошибка ввода. Пример:  1.5 -3.2")
except KeyboardInterrupt:
    print("\nВыход по Ctrl+C")

plt.show()