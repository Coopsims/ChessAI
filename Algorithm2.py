import chess
# Minimax function with alpha-beta pruning.
def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over() or board.is_repetition(3):
        if board.is_repetition(3):
            return 0, None  # return 0 score indicating a draw
        else:
            return evaluation(board, depth), None

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


def evaluation(board, depth):

    depth_score = (depth) * 10

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
        + depth_score
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
        castling_bonus += 1
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_bonus += 1
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_bonus -= 1
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_bonus -= 1
    return castling_bonus