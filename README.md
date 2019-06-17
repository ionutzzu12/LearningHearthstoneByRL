# LearningHearthstoneByRL
Hearthstone Agent training using Reinforcement Learning

This application is designed as an environment for training Hearthstone agents.
The game simulator used is [Spellsource](https://github.com/hiddenswitch/Spellsource-Server). The simulator uses a java server which is running the game logic while playing matches. I have implemented a wrapper for the spellsource python library with the same interface used by OpenAI in its [gym environments](https://gym.openai.com/).
Currently, I am using the [A3C](https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb) algorithm for training a neural network.
