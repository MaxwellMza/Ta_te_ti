import os
import time
from tqdm import tqdm  # Librería para mostrar barras de progreso
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Funciones auxiliares para el juego
def check_winner(board):
    lines = [
        [board[0], board[1], board[2]],
        [board[3], board[4], board[5]],
        [board[6], board[7], board[8]],
        [board[0], board[3], board[6]],
        [board[1], board[4], board[7]],
        [board[2], board[5], board[8]],
        [board[0], board[4], board[8]],
        [board[2], board[4], board[6]]
    ]
    for line in lines:
        if line[0] == line[1] == line[2] and line[0] != '':
            return line[0]
    return None

def is_draw(board):
    return all([spot != '' for spot in board])

def get_available_moves(board):
    return [i for i, spot in enumerate(board) if spot == '']

def minimax(board, is_maximizing):
    winner = check_winner(board)
    if winner == 'X':
        return -1
    elif winner == 'O':
        return 1
    elif is_draw(board):
        return 0

    if is_maximizing:
        best_score = -float('inf')
        for move in get_available_moves(board):
            board[move] = 'O'
            score = minimax(board, False)
            board[move] = ''
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for move in get_available_moves(board):
            board[move] = 'X'
            score = minimax(board, True)
            board[move] = ''
            best_score = min(score, best_score)
        return best_score

def find_best_move(board):
    best_move = None
    best_score = -float('inf')
    for move in get_available_moves(board):
        board[move] = 'O'
        score = minimax(board, False)
        board[move] = ''
        if score > best_score:
            best_score = score
            best_move = move
    return best_move

# Generar datos de entrenamiento
def generate_training_data(num_samples):
    training_data = []
    for _ in tqdm(range(num_samples), desc="Generando datos de entrenamiento"):
        board = [''] * 9
        while True:
            available_moves = get_available_moves(board)
            if not available_moves:
                break
            move = random.choice(available_moves)
            board[move] = 'X'
            if check_winner(board) or is_draw(board):
                break
            best_move = find_best_move(board)
            training_data.append((board.copy(), best_move))
            board[best_move] = 'O'
            if check_winner(board) or is_draw(board):
                break
    return training_data

# Definir función para crear el modelo
def create_model():
    model = Sequential([
        Dense(128, input_dim=9, activation='relu'),
        Dense(128, activation='relu'),
        Dense(9, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Cargar o entrenar el modelo
if os.path.exists('ta_te_ti_model.h5'):
    print("Cargando modelo existente...")
    model = tf.keras.models.load_model('ta_te_ti_model.h5')
    print("Modelo cargado.")
else:
    print("Generando datos de entrenamiento...")
    training_data = generate_training_data(10000)
    print("Datos de entrenamiento generados.")

    # Preparar los datos de entrenamiento
    X = []
    y = []

    for board, move in training_data:
        X.append([1 if spot == 'X' else -1 if spot == 'O' else 0 for spot in board])
        y.append(move)

    X = np.array(X)
    y = np.array(y)

    # Crear y entrenar el modelo
    model = create_model()

    print("Entrenando la IA, por favor espere...")
    start_time = time.time()  # Inicio del temporizador

    # Añadir barra de progreso para el entrenamiento
    class TQDMCallback(tf.keras.callbacks.Callback):
        def __init__(self, epochs):
            self.epochs = epochs
            self.tqdm_bar = tqdm(total=self.epochs, desc="Entrenando la IA")

        def on_epoch_end(self, epoch, logs=None):
            self.tqdm_bar.update(1)

        def on_train_end(self, logs=None):
            self.tqdm_bar.close()

    # Entrenar el modelo
    model.fit(X, y, epochs=10, callbacks=[TQDMCallback(10)])
    end_time = time.time()  # Fin del temporizador
    print(f"Entrenamiento completado en {end_time - start_time:.2f} segundos.")

    # Guardar el modelo entrenado
    model.save('ta_te_ti_model.h5')
    print("Modelo guardado.")

# Función para imprimir el tablero mejorada
def print_board(board):
    """Prints the Ta-Te-Ti board."""
    symbols = {
        'X': '\033[1;31mX\033[m',  # Red color for 'X'
        'O': '\033[1;34mO\033[m',  # Blue color for 'O'
        '': ' '  # Empty spot
    }
    row_divider = '-----------'
    os.system('cls' if os.name == 'nt' else 'clear')  # Limpiar pantalla
    for i in range(3):
        row = [symbols[board[3 * i + j]] for j in range(3)]
        print(' | '.join(row))
        if i < 2:
            print(row_divider)
    print()

# Función para obtener el movimiento de la IA
def get_ai_move(board):
    input_board = np.array([1 if spot == 'X' else -1 if spot == 'O' else 0 for spot in board]).reshape(1, -1)
    prediction = model.predict(input_board)
    best_move = np.argmax(prediction)
    while board[best_move] != '':
        prediction[0][best_move] = -1
        best_move = np.argmax(prediction)
    return best_move

# Función para jugar un solo juego
def play_single_game(starting_player):
    board = [''] * 9
    print_board(board)
    current_player = starting_player

    while True:
        if current_player == 'human':
            # Turno del jugador humano
            human_move = int(input("Ingrese su movimiento (0-8): "))
            if board[human_move] == '':
                board[human_move] = 'X'
            else:
                print("Movimiento inválido. Intente de nuevo.")
                continue
        else:
            # Turno de la IA
            ai_move = get_ai_move(board)
            board[ai_move] = 'O'
            print("IA mueve a:", ai_move)

        print_board(board)

        if check_winner(board) or is_draw(board):
            break

        # Alternar turno
        current_player = 'human' if current_player == 'ai' else 'ai'

    winner = check_winner(board)
    return winner, current_player

# Función para jugar varios juegos
def play_game():
    human_wins = 0
    ai_wins = 0
    draws = 0
    total_games = 6

    starting_player = random.choice(['human', 'ai'])  # Elegir aleatoriamente quién empieza primero

    while True:
        winner, starting_player = play_single_game(starting_player)
        if winner == 'X':
            human_wins += 1
        elif winner == 'O':
            ai_wins += 1
        else:
            draws += 1

        print(f"Puntaje: Humano {human_wins} - IA {ai_wins} - Empates {draws}")

        if human_wins > total_games // 2 or ai_wins > total_games // 2:
            break

    if human_wins > ai_wins:
        print("El ganador final es el Humano.")
    elif ai_wins > human_wins:
        print("El ganador final es la IA.")
    else:
        print("El juego finalizó en empate.")

# Jugar el juego
play_game()
