


import chess
import numpy as np
from joblib import Parallel, delayed


def minimax(board, depth, alpha, beta, maximizingPlayer):
    if depth == 0 or board.is_game_over():
        return evaluation(board)

    if maximizingPlayer:
        maxEval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval
    else:
        minEval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval


def best_move(board, depth):
    moves = list(board.legal_moves)
    boards = [board.copy() for _ in moves]
    for move, b in zip(moves, boards):
        b.push(move)

    # Use Parallel and delayed to create a parallel for loop
    scores = Parallel(n_jobs=-1)(
        delayed(minimax)(b, depth - 1, float('-inf'), float('inf'), not board.turn) for b in boards)

    best_score = max(scores)
    best_move = moves[scores.index(best_score)]

    return best_move


def evaluation(board):

    if board.is_checkmate():
        return -10000000 if board.turn else 10000000

    material = total_material(board)

    # Piece activity and mobility
    piece_map = board.piece_map()
    activity = sum(piece_position_score(piece, pos) for pos, piece in piece_map.items())
    mobility = sum(len(list(board.legal_moves)) for piece in piece_map.values())

    # Control of the center
    center_squares = chess.SquareSet([chess.D4, chess.D5, chess.E4, chess.E5])
    center_control = sum(1 for square in board.piece_map() if square in center_squares)

    # Pawn structure
    pawn_structure = evaluate_pawn_structure(board)

    # King safety and attack
    king_safety = evaluate_king_safety(board, board.turn)

    if not (board.turn):
        activity = -activity
        mobility = -mobility
        pawn_structure = -pawn_structure
        center_control = -center_control
        king_safety = -king_safety


    score = (
        material*4
        + 0.1 * activity
        + 0.05 * mobility
        + 0.4 * center_control
        + 0.2 * pawn_structure
        + 0.4 * king_safety
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

def piece_position_score(piece, position):
    # Piece-square tables
    piece_square_tables = {
        'P': [  # Pawn
            0, 0, 0, 0, 0, 0, 0, 0,
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

    return piece_square_tables[piece.symbol().upper()][position]


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
        return -1000 if board.is_check() else 1000
    else:
        return 1000 if board.is_check() else -1000

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