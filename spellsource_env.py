# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 16:43:21 2018

@author: ioan
"""

from spellsource.behaviour import *
import random
from threading import Thread, Condition, currentThread
import numpy as np

from decks import *


# ENEMY BEHAVIOUR EXAMPLE
class FirstActionBeh(Behaviour):
    def __init__(self, state=None, seed=None):
        pass

    def clone(self):
        return self

    def get_name(self):
        return 'Behaviorel_1stact'

    def mulligan(self, context, player, cards):
        return [c for c in cards if c.getBaseManaCost() > 3]

    def on_game_over(self, context, playerId, winningPlayerId):
        pass

    def request_action(self, context, player, valid_actions):
        return valid_actions[0]


def print_action(action, game_context):
    i = game_context.getActivePlayerId()
    source = game_context.resolveSingleTarget(action.getSourceReference())
    target = game_context.resolveSingleTarget(action.getTargetReference())
    source = None if source is None else source.getName()
    target = None if target is None else target.getName()
    print('%d %s %s: Targeting %s' % (i, action.getActionType().toString(), source, target))


def print_trace(game_context):
    for action in game_context.getTrace().getRawActions():
        action_type = action.getActionType().toString()
        source = action.getSource(game_context)
        player_id = source.getOwner()
        targets = action.getTargets(game_context, player_id)
        print(player_id,
              action_type,
              source.getName() if source is not None else "(no source)",
              'ON' if len(targets) > 0 else '',
              ', '.join(target.getName() for target in targets))


class AIBehaviour(Behaviour):
    def __init__(self, condition1, condition2):
        self.valid_actions = None
        self.chosen_action = None
        self.condition1 = condition1
        self.condition2 = condition2
        self.winner = None
        self.context = None
        self.encoded_context = None
        self.current_player = None

    def _get_object_id(self):
        return 'AI0'

    def clone(self):
        return self

    def get_name(self):
        return 'AI0'

    def mulligan(self, context, player, cards):
        return [c for c in cards if c.getBaseManaCost() > 3]

    def on_game_over(self, context, playerId, winningPlayerId):
        self.context = context
        self.encoded_context = None
        self.current_player = playerId
        self.winner = (playerId, winningPlayerId)

        condition1 = self.condition1
        valid_actions = []  # signals endgame

        condition1.acquire()

        if self.valid_actions is not None:
            condition1.wait()
        self.valid_actions = valid_actions
        condition1.notify()
        condition1.release()

    def request_action(self, context, player, valid_actions):
        # print(len(valid_actions), 'actions in behaviour')
        # for a in valid_actions: print_action(a, context)
        self.context = context  # useful in env
        # TODO: change encoder more easily
        self.encoded_context = np.array(list(context.java_encode_state_v3_updated_with_many_stats()))
        self.current_player = player.getId()

        condition1 = self.condition1
        condition2 = self.condition2

        condition1.acquire()

        if self.valid_actions is not None:
            condition1.wait()
        self.valid_actions = valid_actions
        condition1.notify()
        condition1.release()

        condition2.acquire()
        if self.chosen_action is None:
            condition2.wait()
        assert self.chosen_action is not None
        return_action = valid_actions[self.chosen_action]
        self.chosen_action = None

        condition2.notify()
        condition2.release()
        assert return_action is not None
        return return_action


class SpellsourceEnv:
    @staticmethod
    def init():
        return SpellsourceEnv()

    def get_rollout_states(self, actions):
        assert not self.done
        assert self.my_beh.context is not None
        context = self.my_beh.context
        encoded_states = [context.apply_action_and_encode(a) for a in actions]
        return encoded_states  # sometimes the server explodes

    def __init__(self):
        self.ctx = Context()
        self.condition1 = Condition()
        self.condition2 = Condition()
        self.match_count = self.wins = self.losses = 0
        self.done = True
        self.valid_actions = None
        self.my_beh = AIBehaviour(condition1=self.condition1, condition2=self.condition2)

        self.DECK1_TO_USE = ZOO_WARLOCK
        self.DECK2_TO_USE = ZOO_WARLOCK

    def process_match_and_get_reward(self):
        reward = 0
        self.done = True
        self.match_count += 1
        my_id, winner_id = self.my_beh.winner
        if winner_id == 1:  # won
            reward = 1
            self.wins += 1
        elif winner_id == 0:  # lost
            reward = -1
            self.losses += 1
        else:
            print('in process_reward.. there is actually a TIE: r:', reward, 'crt_player:', my_id, 'winner:', winner_id)
        return reward

    def wait_for_valid_actions(self):
        self.condition1.acquire()
        if self.my_beh.valid_actions is None:
            self.condition1.wait()
        assert self.my_beh.valid_actions is not None
        self.valid_actions = self.my_beh.valid_actions
        self.my_beh.valid_actions = None
        self.condition1.notify()
        self.condition1.release()

    def get_encoded_state(self):
        return self.my_beh.encoded_context

    def reset(self, opponent='random', disable_fatigue=True):
        game_context = self.ctx.GameContext.fromDeckLists([self.DECK1_TO_USE, self.DECK2_TO_USE])
        # to set another behaviour (index 0 is enemy)
        # TODO: default is random now
        if opponent.lower() != 'random':
            if opponent.lower() == 'gsvb':
                game_context.setGSVB(0)
            elif opponent.lower() == 'firstaction':
                game_context.setBehaviour(0, FirstActionBeh().wrap(self.ctx))
            elif opponent.lower() == 'greedy':
                game_context.setGreedy(0)
            elif opponent.lower() == 'mcts':
                game_context.setMCTS(0)
            elif opponent.lower() == 'self':
                # TODO: playing versus itself, new Conditions for better synchronization
                self.condition1 = Condition()
                self.condition2 = Condition()
                self.my_beh = AIBehaviour(condition1=self.condition1, condition2=self.condition2)
                game_context.setBehaviour(0, self.my_beh.wrap(self.ctx))

        # 1 is me
        game_context.setBehaviour(1, self.my_beh.wrap(self.ctx))
        work = lambda: game_context.play()
        t = Thread(target=work)
        t.start()
        self.done = False

        # get first set of valid_actions
        self.wait_for_valid_actions()

        reward = 0
        assert self.valid_actions
        if disable_fatigue:
            game_context.disable_fatigue()
        return self.get_encoded_state(), reward, self.done, self.valid_actions, self.my_beh.current_player

    def step(self, action):
        assert not self.done
        assert action is not None
        assert action in range(len(self.valid_actions))

        # set action
        self.condition2.acquire()
        if self.my_beh.chosen_action is not None:
            self.condition2.wait()
        assert self.my_beh.chosen_action is None
        self.my_beh.chosen_action = action
        self.condition2.notify()
        self.condition2.release()

        # get next set of valid_actions
        self.wait_for_valid_actions()

        reward = 0
        if not self.valid_actions:  # empty
            reward = self.process_match_and_get_reward()
            assert self.done
        return self.get_encoded_state(), reward, self.done, self.valid_actions, self.my_beh.current_player

    def save_replay(self, filename):
        assert self.done
        with open(filename, 'w') as f:
            game_context = self.my_beh.context
            f.write('winner: ' + str(game_context.getWinningPlayerId()) + '\n')
            for action in game_context.getTrace().getRawActions():
                action_type = action.getActionType().toString()
                source = action.getSource(game_context)
                player_id = source.getOwner()
                targets = action.getTargets(game_context, player_id)
                f.write('\n' + str(player_id) + ' ' +
                        str(action_type) + ' ' +
                        str(source.getName() if source is not None else "(no source)") +
                        (' ON ' if len(targets) > 0 else '') +
                        ', '.join((target.getName() + '(' + str(target.getOwner()) + ')') for target in targets))
            f.close()


class Tester(Thread):  # main
    def run(self):
        env = SpellsourceEnv.init()
        print(currentThread().getName(), 'start match 1')
        total_r = 0
        d = {}

        while True:
            enc_state, r, done, actions, player = env.reset('random')
            context = env.my_beh.context
            # game configurations
            # context.set_precompute_rollouts()
            # context.disable_fatigue()
            total_r += r

            while not done:
                num_acts = len(actions)
                if num_acts in d:
                    d[num_acts] += 1
                else:
                    d[num_acts] = 1

                # TODO: testing speed
                c = env.my_beh.context
                """
                print(len(c.getValidActions()), 'actions in context')
                for a in c.getValidActions(): print_action(a, c)
                print(len(actions),'actions here')
                for a in actions: print_action(a, c)
                
                # st_list = env.get_rollout_states(actions)
                st_list = env.my_beh.context.get_rollout_encoded_states()
                # print(st_list)

                while st_list is None:
                    st_list = env.my_beh.context.compute_rollout_encoded_states()
                # print(st_list)
                # print(len(st_list))
                arr = np.array(list(st_list))
                if not len(st_list) == len(actions):
                    print('wtf')
                # print(st_list)
                # print('action2')
                # assert player == 1
                # print(player, enc_state)
                """
                a = random.choice(range(len(actions)))
                # print(actions.toString())
                # arr = np.array(list(c.java_encode_state_v4_ids_one_hot()))
                # assert actions[-1].toString() == "[END_TURN]"
                # assert len(actions) == len(c.getValidActions())
                enc_state, r, done, actions, player = env.step(a)
                total_r += r

            print('match!', r)
            # print_trace(env.my_beh.context)
            if env.match_count % 10 == 0:
                x = max(d.keys())
                if x == 81:
                    print([a.toString() for a in actions])
                # print(currentThread().getName(), env.match_count, x)
                print(currentThread().getName(), env.match_count, env.wins, x)
                assert total_r == env.wins - env.losses


if __name__ == "__main__":
    for num_threads in range(1):
        Tester().start()
