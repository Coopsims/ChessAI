


import chess
import numpy as np
import tensorflow as tf
import time
from concurrent.futures import ProcessPoolExecutor


def iterative_deepening_best_move(board, start_time, max_depth):
    best_move = None
    best_score = float('-inf')  # Initialize best_score to negative infinity
    finalDepth=0
    for depth in range(1, max_depth + 1):
        if time.time() - start_time < 40.0:

            # Calculate the current board score here
            current_board_score = evaluation(board)/100
            best_move_at_current_depth = best_move_at_depth(board, start_time, depth, current_board_score)

            if best_move_at_current_depth:  # Check if a valid move was returned
                # Re-calculate the score for this specific best move
                temp_board = board.copy()
                temp_board.push(best_move_at_current_depth)
                new_score = evaluation(temp_board)/100

                if new_score > best_score:  # Update best_move and best_score if a better move is found
                    best_move = best_move_at_current_depth
                    best_score = new_score
                    finalDepth = depth

    print(f"Depth: {finalDepth}, Best Move: {best_move}, Score: {best_score: .3f}")

    return best_move


def best_move_at_depth(board, start_time, depth, current_board_score):
    moves = list(board.legal_moves)
    boards = [board.copy() for _ in moves]
    for move, b in zip(moves, boards):
        b.push(move)

    alpha = float('-inf')
    beta = float('inf')
    maximizingPlayer = True

    with ProcessPoolExecutor() as executor:
        scores = list(
            executor.map(
                perform_minimax,
                boards,
                [alpha] * len(boards),
                [beta] * len(boards),
                [maximizingPlayer] * len(boards),
                [depth] * len(boards),
                [start_time] * len(boards),
                [current_board_score] * len(boards)
            )
        )

    valid_moves_and_scores = [pair for pair in zip(moves, scores) if pair[1] is not None]
    if not valid_moves_and_scores:
        return None  # Fallback logic can be added here

    best_score = max(valid_moves_and_scores, key=lambda x: x[1])[1]
    best_move = [m for m, s in valid_moves_and_scores if s == best_score][0]
    return best_move


def perform_minimax(board, alpha, beta, maximizingPlayer, depth, start_time, current_board_score):
    return minimax(board, alpha, beta, maximizingPlayer, depth, start_time, current_board_score)

def minimax(board, alpha, beta, maximizingPlayer, depth, start_time, current_board_score):
    if time.time() - start_time >= 40.0 or depth == 0:
        score = evaluation(board)
        return score/100.0 if score >= current_board_score - 1 else None

    if maximizingPlayer:
        maxEval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, alpha, beta, False, depth-1, start_time, current_board_score)
            board.pop()
            if eval is None:
                continue
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval if maxEval != float('-inf') else float('-inf')
    else:
        minEval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, alpha, beta, True, depth-1, start_time, current_board_score)
            board.pop()
            if eval is None:
                continue
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval if minEval != float('inf') else float('inf')

def evaluation(board):

    if board.is_checkmate():
        return -10000000 if board.turn else 10000000

    material = total_material(board)

    # Piece activity and mobility
    piece_map = board.piece_map()
    activity = sum(piece_position_score(piece, pos, board.turn) for pos, piece in piece_map.items())
    mobility = sum(len(list(board.legal_moves)) for piece in piece_map.values())

    # Control of the center
    center_squares = chess.SquareSet([chess.D4, chess.D5, chess.E4, chess.E5])
    center_control = sum(1 for square in board.piece_map() if square in center_squares)

    # Pawn structure
    pawn_structure = evaluate_pawn_structure(board)

    # King safety and attack
    king_safety = evaluate_king_safety(board, board.turn)

    if not board.turn:
        material = -material
        activity = -activity
        mobility = -mobility
        pawn_structure = -pawn_structure
        center_control = -center_control

    score = (
            material
            + 0.1 * activity
            + 0.1 * mobility
            + 0.4 * center_control
            + 0.2 * pawn_structure
            + 0.6 * king_safety
    )

    return score

def total_material(board):
    piece_values = {chess.PAWN: 100, chess.KNIGHT: 280, chess.BISHOP: 320, chess.ROOK: 500, chess.QUEEN: 900}

    white_score = 0
    black_score = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_value = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                white_score += piece_value
            else:
                black_score += piece_value

    return white_score - black_score

def is_endgame(board):
    total_materials = total_material(board)

    # Adjust this threshold to fine-tune when the function considers the game to be in the endgame phase
    endgame_threshold = 1000

    return total_materials <= endgame_threshold

def piece_position_score(piece, position, is_white_turn):
    # Piece-square tables
    piece_square_tables = {
        'P': [  # Pawn
            100, 100, 100, 100, 100, 100, 100, 100,
            5, 10, 10, -20, -20, 10, 10, 5,
            5, -5, -10, 0, 0, -10, -5, 5,
            1, 1, 1, 40, 40, 1, 1, 1,
            5, 5, 10, 55, 55, 10, 5, 5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 30, 30, 50, 50, 50,
            0, 0, 0, 0, 0, 0, 0, 0],
        'N': [  # Knight
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -30, 5, 15, 20, 20, 15, 5, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 5, 30, 15, 15, 30, 5, -30,
            -40, -20, 0, 5, 5, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50],
        'B': [  # Bishop
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 10, 10, 5, 0, -10,
            -10, 5, 5, 10, 10, 5, 5, -10,
            -10, 0, 10, 10, 10, 10, 0, -10,
            -10, 10, 10, 10, 10, 10, 10, -10,
            -10, 5, 0, 0, 0, 0, 5, -10,
            -20, -10, -10, -10, -10, -10, -10, -20],
        'R': [  # Rook
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, 10, 10, 10, 10, 5,
            -5, 1, 1, 1, 1, 1, 1, -5,
            -5, 1, 1, 1, 1, 1, 1, -5,
            -5, 1, 1, 1, 1, 1, 1, -5,
            -5, 1, 1, 1, 1, 1, 1, -5,
            -5, 1, 1, 1, 1, 1, 1, -5,
            1, 1, 10, 10, 10, 10, 1, 1],
        'Q': [  # Queen
            -20, -10, -10, -5, -5, -10, -10, -20,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -10, 0, 5, 5, 5, 5, 0, -10,
            -5, 0, 5, 5, 5, 5, 0, -5,
            0, 0, 5, 5, 5, 5, 0, -5,
            -10, 5, 5, 5, 5, 5, 0, -10,
            -10, 0, 5, 0, 0, 0, 0, -10,
            -20, -10, -10, -5, -5, -10, -10, -20],
        'K': [  # King
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -20, -30, -30, -40, -40, -30, -30, -20,
            -10, -20, -20, -20, -20, -20, -20, -10,
            20, 20, 0, 0, 0, 0, 20, 20,
            20, 50, 10, 0, 0, 40, 50, 20]
    }

    if is_white_turn:
        return piece_square_tables[piece.symbol().upper()][position]
    else:
        # Flip the board for black
        flipped_position = 63 - position
        return -piece_square_tables[piece.symbol().upper()][flipped_position]


def file_of(square):
    return square % 8

def rank_of(square):
    return square // 8


def evaluate_pawn_structure(board):
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None and piece.piece_type == chess.PAWN:
            # Check for isolated pawns
            isolated = not any(board.piece_at(chess.square(file, rank_of(square))).piece_type == chess.PAWN
                               for file in [max(0, file_of(square) - 1), min(7, file_of(square) + 1)]
                               if board.piece_at(chess.square(file, rank_of(square))) is not None)
            if isolated:
                score -= 30
            # Check for doubled pawns
            doubled = any(board.piece_at(chess.square(file_of(square), rank)).piece_type == chess.PAWN
                          for rank in range(rank_of(square))
                          if board.piece_at(chess.square(file_of(square), rank)) is not None)
            if doubled:
                score -= 40
    return score

def evaluate_king_safety(board, color):
    if color==chess.WHITE:
        return -500 if board.is_check() else 500
    else:
        return 500 if board.is_check() else -500

def evaluate_castling(board):
    castling_bonus = 0
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_bonus += 10
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_bonus += 10
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_bonus -= 10
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_bonus -= 10
    return castling_bonus