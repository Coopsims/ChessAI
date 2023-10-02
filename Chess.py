import csv
import uuid
import chess
import algorithm1 as algor
import Algorithm2 as algor2
import chess.polyglot
import time
def main():
    learning = 0
    while learning <= 100:
        board = chess.Board()
        initial_board = board.copy()  # Copy the initial board state here
        move_list = []
        check_evaluation_polarities()
        while not board.is_checkmate() and not board.is_stalemate():
            if board.turn:
                move2 = algor.iterative_deepening_best_move(board, time.time(), max_depth=20)  # Adjust the depth as needed
                if move2 is None:
                    break
                move_list.append(board.san(move2))
                board.push(move2)
                print(f"White Move: {move2}")
                print(board)
                if board.is_repetition(3):
                    print("Draw due to threefold repetition!")
                    break
            else:
                while True:
                    move1_san = input("Enter your move: ")  # Ask for user input
                    try:
                        move1 = board.parse_san(move1_san)
                        if move1 not in board.legal_moves:
                            print("Invalid move, try again.")
                            continue
                        break
                    except:
                        print("Invalid input, try again.")
                move_list.append(move1_san)
                board.push(move1)

                print(f"Black move: {move1_san}")
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


def check_evaluation_polarities():
    # Let's use a start position for our board
    board = chess.Board()

    # Evaluate the board from white's perspective
    white_evaluation = algor.evaluation(board)

    # Now, flip the board to black's perspective
    board.turn = chess.BLACK
    black_evaluation = algor.evaluation(board)

    # Perform checks on the evaluation function

    # Material
    if algor.total_material(board) != -(white_evaluation - black_evaluation):
        print("Polarity error in material evaluation.")

if __name__ == "__main__":
    main()
