# 成三棋

## 规则

1．游戏分两个阶段――下棋阶段和走棋阶段。在第一阶段（下棋阶段）下棋双方各有9颗棋子，将棋子下在线段交叉处。轮流下完各自的棋子后，自动进入下一阶段（走棋阶段）。
2．在下棋和走棋的过程中，任何一方有三颗棋子连续排列成一条直线时，则成三了，这是可以揪掉对方的一个棋子。但注意在揪棋子的时候不能揪对方已经成三的棋子
3．在下棋的过程中，被揪掉棋子的位置将不能再下棋子（该位置会画上红×，直到走棋阶段才能放棋子）。
4．在走棋的过程中，一次只能走一个单位（起点到终点之间不能有交叉点）。
5．被揪棋的一方，在被揪棋后只剩下2颗棋子或更少，则被揪方输了这一局。

## 游戏结束条件 (Game End Conditions)

游戏在以下情况下结束：

1. 在走棋阶段，如果一方棋子数量少于3个，该方输掉游戏。
2. 在走棋阶段，如果轮到一方走棋时无法移动任何棋子（被对方封死），该方输掉游戏。
3. 如果游戏超过100步仍未分出胜负，则判定为平局。

Game ends when one of these conditions is met:

1. During the movement phase, if a player has fewer than 3 pieces, they lose the game.
2. During the movement phase, if a player has no valid moves on their turn (completely blocked), they lose the game.
3. If the game exceeds 100 moves without a winner, it is declared a draw.

## 胜负判定 (Win/Loss Determination)

1. 胜利条件：
   - 将对方棋子减少到少于3个
   - 封锁对方所有可能的移动

2. 失败条件：
   - 己方棋子少于3个
   - 无法进行任何合法移动

3. 平局条件：
   - 游戏步数超过100步
   - 双方都无法形成获胜局面

## 棋盘 (Game Board)

The Three Men's Morris board consists of three concentric squares with connecting lines: 

When a mill is formed, you can remove one of your opponent's pieces that is not part of a mill.

2. Vertical mill:

```
 •───────────X───────────•
 │             │             │
 │    •───────X───────•    │
 │    │        │        │    │
 │    │   •───X───•   │    │
 │    │   │         │   │    │
 •────•───•         •───•────•
 │    │   │         │   │    │
 │    │   •───O───•   │    │
 │    │        │        │    │
 │    •───────O───────•    │
 │             │             │
 •───────────•───────────•
```

When a mill is formed, you can remove one of your opponent's pieces that is not part of a mill.

## 成三 (Mill Formation)

A mill is formed when three pieces of the same player are in a row. Examples:

1. Horizontal mill:

```
 •───────────•───────────•
 │             │             │
 │    X───────X───────X    │
 │    │        │        │    │
 │    │   •───•───•   │    │
 │    │   │         │   │    │
 •────•───•         •───•────•
 │    │   │         │   │    │
 │    │   •───•───•   │    │
 │    │        │        │    │
 │    O───────•───────O    │
 │             │             │
 •───────────•───────────•
```

2. Vertical mill:

```
 •───────────X───────────•
 │             │             │
 │    •───────X───────•    │
 │    │        │        │    │
 │    │   •───X───•   │    │
 │    │   │         │   │    │
 •────•───•         •───•────•
 │    │   │         │   │    │
 │    │   •───O───•   │    │
 │    │        │        │    │
 │    •───────O───────•    │
 │             │             │
 •───────────•───────────•
```

When a mill is formed, you can remove one of your opponent's pieces that is not part of a mill.

Example of pieces on the board:

```
 X───────────•───────────O
 │             │             │
 │    •───────X───────•    │
 │    │        │        │    │
 │    │   •───•───•   │    │
 │    │   │         │   │    │
 •────•───•         •───•────•
 │    │   │         │   │    │
 │    │   O───•───•   │    │
 │    │        │        │    │
 │    •───────O───────•    │
 │             │             │
 •───────────•───────────X
```

## 棋子 (Game Pieces)

Players use X and O as their pieces:
