
ZOO_SMALL = '''### D1
# Class: Warlock
# Format: Standard
# Year of the Raven
#
# 2x (1) Flame Imp
# 2x (1) Abusive Sergeant
# 2x (3) Harvest Golem
# 2x (5) Doomguard
# 2x (12) Mountain Giant
#
'''

ZOO_1_MINION = '''### D1
# Class: Warlock
# Format: Standard
# Year of the Raven
#
# 2x (1) Abusive Sergeant
# 2x (1) Flame Imp
#
'''

DECK_priest_small = '''### D2
# Class: Priest
# Format: Standard
# Year of the Raven
#
# 2x (0) Circle of Healing
# 2x (1) Inner Fire
# 2x (2) Divine Spirit
# 2x (2) Lightwell
# 2x (3) Shadow Word: Death
# 2x (4) Stormwind Knight
# 2x (4) Twilight Drake
#
'''

ZOO_WARLOCK = '''### ZOO_WARLOCK
# Class: Warlock
# Format: Standard
# Year of the Raven
#
# 2x (1) Abusive Sergeant
# 2x (1) Argent Squire
# 2x (1) Voodoo Doctor
# 2x (1) Voidwalker
# 2x (1) Young Priestess
# 2x (1) Flame Imp
# 2x (1) Soulfire
# 2x (2) Amani Berserker
# 2x (2) Dire Wolf Alpha
# 2x (2) Knife Juggler
# 2x (3) Harvest Golem
# 2x (3) Shattered Sun Cleric
# 2x (4) Dark Iron Dwarf
# 2x (4) Defender of Argus
# 2x (5) Doomguard
#
'''

AGGRO_PALADIN = '''### AGGRO_PALADIN
# Class: Paladin
# Format: Standard
# Year of the Raven
#
# 2x (1) Blessing of Might
# 2x (1) Abusive Sergeant
# 2x (1) Arcane Anomaly
# 2x (1) Argent Squire
# 2x (1) Runic Egg
# 2x (1) Worgen Infiltrator
# 2x (2) Argent Protector
# 2x (2) Acidic Swamp Ooze
# 2x (2) Knife Juggler
# 2x (3) Divine Favor
# 2x (3) Shattered Sun Cleric
# 2x (3) Wolfrider
# 2x (4) Blessing of Kings
# 2x (4) Consecration
# 2x (5) Truesilver Champion
#
'''

EVEN_PALADIN = '''### EVEN_PALADIN
# Class: Paladin
# Format: Standard
#
# 1x Acidic Swamp Ooze
# 2x Amani Berserker
# 1x Crystalsmith Kangor
# 2x Equality
# 1x Hydrologist
# 2x Wild Pyromancer
# 2x Blessing of Kings
# 2x Consecration
# 2x Corpsetaker
# 2x Saronite Chain Gang
# 1x The Glass Knight
# 2x Truesilver Champion
# 1x Avenging Wrath
# 1x Genn Greymane
# 1x Mossy Horror
# 2x Spikeridged Steed
# 1x Sunkeeper Tarim
# 1x Val'anyr
# 1x Windfury Harpy
# 1x Dinosize
# 1x The Lich King
#
'''


if __name__ == "__main__":
    from spellsource.context import Context

    with Context() as ctx:
        wins = 0
        total = 30

        for i in range(total):
            game_context = ctx.GameContext.fromDeckLists([ZOO_WARLOCK, EVEN_PALADIN])
            game_context.setGSVB(0)
            game_context.setGSVB(1)
            print('Starting match..', i)
            game_context.play()
            # print('Finish! Winner is', game_context.getWinningPlayerId())

            if game_context.getWinningPlayerId() == 1:
                wins += 1

        print('win rate:', wins/total)
