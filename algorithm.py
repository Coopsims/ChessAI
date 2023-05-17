import chess
# Minimax function with alpha-beta pruning.
def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluation(board), None

    board_hash = hash(str(board))
    if board_hash in transposition_table:
        return transposition_table[board_hash], None

    if maximizing_player:
        max_value = float('-inf')
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            eval_child, _ = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            if eval_child > max_value:
                max_value = eval_child
                best_move = move
            alpha = max(alpha, eval_child)
            if beta <= alpha:
                break
        transposition_table[board_hash] = max_value
        return max_value, best_move
    else:
        min_value = float('inf')
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            eval_child, _ = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            if eval_child < min_value:
                min_value = eval_child
                best_move = move
            beta = min(beta, eval_child)
            if beta <= alpha:
                break
        transposition_table[board_hash] = min_value
        return min_value, best_move


# Function to get the best move for the current player.
transposition_table = {}

def best_move(board, max_depth):
    global transposition_table
    transposition_table = {}  # Clear the transposition table

    best_move = None
    best_value = float('-inf') if board.turn else float('inf')

    for depth in range(1, max_depth + 1):
        value, move = minimax(board, depth, float('-inf'), float('inf'), board.turn)
        if board.turn and value > best_value:
            best_value = value
            best_move = move
        elif not board.turn and value < best_value:
            best_value = value
            best_move = move

    print(best_move, best_value)
    return best_move


def evaluation(board):
    if board.is_checkmate():
        # If the current player (the one the AI is playing as) is in checkmate, this is very bad
        if board.turn:
            return -100000
        # If the opponent is in checkmate, this is very good
        else:
            return 100000

    material = total_material(board, chess.WHITE) - total_material(board, chess.BLACK)

    # Capturing bonus
    capture_bonus = 200 * (total_material(board, chess.BLACK) - total_material(board, chess.WHITE))

    # Piece activity
    activity = sum(piece_position_score(piece, pos) for pos, piece in board.piece_map().items())

    # Control of the center
    center_squares = chess.SquareSet([chess.D4, chess.D5, chess.E4, chess.E5])
    center_control = sum(1 for square in board.piece_map() if square in center_squares)

    # Pawn structure
    pawn_structure = evaluate_pawn_structure(board)

    # King safety
    king_safety = evaluate_king_safety(board, board.turn)


    # Combine the factors to get a final score
    score = material + 0.1 * activity + 0.1 * center_control + 0.2 * pawn_structure + 0.4 * king_safety+capture_bonus

    # Adjust the score based on the current player
    if board.turn:

        return score
    else:

        return -score

def total_material(board, color):
    piece_values = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000}

    return sum(piece_values.get(piece.symbol().upper(), 0) for piece in board.piece_map().values() if piece.color == color)


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
    if piece.color == chess.WHITE:
        return -piece_square_tables[piece.symbol().upper()][position]
    else:
        return piece_square_tables[piece.symbol().upper()][63 - position]

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
                score -= 10
            # Check for doubled pawns
            doubled = any(board.piece_at(chess.square(file_of(square), rank)).piece_type == chess.PAWN
                          for rank in range(rank_of(square))
                          if board.piece_at(chess.square(file_of(square), rank)) is not None)
            if doubled:
                score -= 20
    return score

def evaluate_king_safety(board, color):
    if color==chess.WHITE:
        return -100 if board.is_check() else 100
    else:
        return 100 if board.is_check() else -100

def evaluate_castling(board):
    castling_bonus = 0
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_bonus += 1
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_bonus += 1
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_bonus -= 1
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_bonus -= 1
    return castling_bonus