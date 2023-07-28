import csv
import os
import chess
import random
import algorithm1 as algor
import Algorithm2 as algor2
import chess.polyglot
import AI
# Initialize an empty chess board

import csv
import uuid

learning = 0
while learning <= 100:
    board = chess.Board()
    initial_board = board.copy()  # Copy the initial board state here
    move_list = []
    while not board.is_checkmate() and not board.is_stalemate():
        if board.turn:
            move1 = algor.best_move(board, 2)  # Adjust the depth as needed
            if move1 is None:
                break
            move_list.append(board.san(move1))
            board.push(move1)

            print(f"White move: {move1}")
            print(board)
            if board.is_repetition(3):
                print("Draw due to threefold repetition!")
                break
        else:
            move2 = algor.best_move(board, 2)  # Adjust the depth as needed
            if move2 is None:
                break
            move_list.append(board.san(move2))
            board.push(move2)
            print(f"Black move: {move2}")
            print(board)
            if board.is_repetition(3):
                print("Draw due to threefold repetition!")
                break

            print(move_list)
    if board.is_checkmate():
        if board.turn:
            print("Black wins by checkmate!")
        else:
            print("White wins by checkmate!")


    filename = f"/Users/benfunk/PycharmProjects/pythonProject/pastGames/{uuid.uuid4()}.csv"
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Writing each move on a new row
        for move in move_list:
            writer.writerow([move])

    print(f"Saved move list to {filename}")
    learning = 101

