# 🐸 Eddie Zuma World 🌈

Welcome to Eddie Zuma World, a colorful and challenging ball-shooting game! In this magical world, you'll play as Eddie, a brave frog who uses his precise tongue-shooting skills to eliminate advancing colorful balls. Are you ready for the challenge? Let's dive into Eddie's vibrant world!

## 🎮 Game Features

- 🌀 Spiraling queue of colorful balls
- 🎯 360-degree rotating shooting system
- 🧠 Intelligent AI opponent
- 🔥 Dynamic difficulty adjustment
- 🏆 Real-time scoring system

## 🛠 Tech Stack

- 🐍 Python 3.x
- 🎨 Pygame - Game graphics and sound
- 🤖 PyTorch - AI model training
- 🔌 Socket - Network communication
- 🧵 Threading - Multi-threaded processing

## 🚀 Quick Start

1. Clone the repository:
   ```
   https://gitlink.org.cn/eddie2001/ZumaWorld.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Start the server:
   ```
   python zuma.py
   ```

4. Launch the client:
   ```
   python client.py
   ```

5. (Optional) Start the AI client:
   ```
   python rl_client.py
   ```

## 🕹 Controls

- ⬅️ Left Arrow Key: Rotate Eddie counter-clockwise
- ➡️ Right Arrow Key: Rotate Eddie clockwise
- 🚀 Spacebar: Shoot a colored ball

## 🧠 AI Mode

Want to challenge yourself further? Try our AI mode! The AI uses deep reinforcement learning algorithms to autonomously learn and continuously improve its game skills. Launch `rl_client.py` to watch the AI in action or compete against it!


## 🧠 AI Logic

Our AI client (`rl_client.py`) uses Deep Q-Learning (DQN), a powerful reinforcement learning algorithm, to master Eddie Zuma World. Here's a breakdown of the AI logic:

1. **State Representation**: 
   - The game state is represented as a 7-dimensional vector, including:
     - Shooter angle (normalized)
     - Shooter color (encoded)
     - Number of balls on the field
     - Current score
     - Game over status
     - Current difficulty level
     - Path length

2. **Action Space**:
   - The AI can perform 3 actions:
     - Rotate left
     - Rotate right
     - Shoot

3. **Neural Network**:
   - A 3-layer fully connected neural network (DQN) is used to approximate the Q-function.
   - Input: 7-dimensional state vector
   - Hidden layers: 2 layers with 128 neurons each
   - Output: Q-values for each of the 3 possible actions

4. **Exploration vs Exploitation**:
   - Epsilon-greedy strategy is employed for action selection.
   - The AI starts with high exploration (epsilon = 1.0) and gradually shifts towards exploitation (epsilon_min = 0.01).

5. **Experience Replay**:
   - A replay buffer stores past experiences (state, action, reward, next_state, done).
   - The AI learns from random batches of these experiences, breaking correlations between consecutive samples.

6. **Training Process**:
   - The AI updates its policy every 4 steps.
   - It uses a separate target network, updated every 1000 steps, for more stable learning.

7. **Reward Function**:
   - Rewards are calculated based on:
     - Score increase
     - Reduction in the number of balls on the field
     - Penalty for game over

8. **Continuous Learning**:
   - The AI client runs in a loop, continuously playing games and updating its policy.
   - The model is saved periodically, allowing for persistent improvement over time.

This sophisticated AI system allows Eddie to learn and adapt to the game dynamics, continuously improving its performance and providing an ever-evolving challenge for human players!


## 🤝 Contributing

Issues and feature requests are welcome! If you'd like to contribute to Eddie Zuma World, please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

Special thanks to all the developers and players who have contributed to Eddie Zuma World. It's because of you that Eddie's world is so amazing!

Are you ready? Let's join Eddie's adventure and become the champion of the Zuma world! 🏆🐸
