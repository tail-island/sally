import gym
import numpy as np


learning_rate = 0.1  # 学習率。0〜1。0に近いと学習は遅いけど精度が高く、1に近いと学習は速いけど精度が低くなります。
discount = 0.9       # 割引率。0〜1。0に近いと直近の結果を重視、1に近いと将来の結果を重視するようになります。

env = gym.make('FrozenLake-v0')
env._max_episode_steps = 10000  # FrozenLake-v0は100手で終了してしまう（TimeLimitでラップしてある）ので、制限を緩めておきます。

q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)


def train():
    for _ in range(5000):
        observation = env.reset()
        done = False

        while not done:
            action = np.argmax(q_table[observation]) if np.random.random() < 0.9 else env.action_space.sample()  # 基本はQテーブルからだけど、時々ランダムでアクションを決定します。

            next_observation, reward, done, _ = env.step(action)  # アクションを実行します。

            if done and reward == 0:  # FrozenLake-v0はゴールに辿り着いた時の報酬が1でそれ以外は0なので、行動を避ける方向の学習が働きません。
                reward = -1           # なので、終了 and 報酬 == 0のとき（穴に落ちた or TimeLimitに引っかかったとき）は、報酬を-1にしておきます。

            q_table[observation, action] += learning_rate * (reward + discount * np.max(q_table[next_observation]) - q_table[observation, action])  # Qテーブルを更新します。

            observation = next_observation


def check():
    observation = env.reset()
    done = False

    while not done:
        env.render()

        action = np.argmax(q_table[observation])
        observation, reward, done, _ = env.step(action)

    env.render()


if __name__ == '__main__':
    train()
    check()

    def evaluate(action_f):
        total_reward = 0

        observation = env.reset()
        done = False

        while not done:
            action = action_f(observation)
            observation, reward, done, _ = env.step(action)

            total_reward += reward

        return total_reward

    print(np.mean(tuple(map(lambda _: evaluate(lambda observation: np.argmax(q_table[observation])), range(1000)))))
    print(np.mean(tuple(map(lambda _: evaluate(lambda observation: {0: 0, 1: 3, 2: 3, 3: 3, 4: 0, 6: 2, 8: 3, 9: 1, 10: 0, 13: 2, 14: 1}[observation]), range(1000)))))  # たぶんこれが模範解答。
    print(q_table)
