# =============================================================================
# snake_game.py — Environment game Snake menggunakan Pygame
#
# Interface utama untuk agent RL:
#   - reset()           → reset game ke state awal, return state
#   - play_step(action) → jalankan satu aksi, return (reward, done, score)
#   - get_state()       → return state vector 11 dimensi
#
# Aksi menggunakan format RELATIF terhadap arah gerak saat ini:
#   [1, 0, 0] = lurus
#   [0, 1, 0] = belok kanan
#   [0, 0, 1] = belok kiri
# =============================================================================

import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

import sys
import os

# Tambahkan root directory ke path agar bisa import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

# --- Konstanta ---
# Warna menggunakan RGB tuple
WHITE      = (255, 255, 255)
BLACK      = (0,   0,   0)
RED        = (200, 0,   0)
RED_LIGHT  = (255, 100, 100)
GREEN      = (0,   180, 0)
GREEN_DARK = (0,   140, 0)
GRAY       = (40,  40,  40)
BLUE       = (0,   100, 200)

# Font untuk score display
pygame.font.init()
FONT = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    """Empat arah absolut yang mungkin untuk pergerakan ular."""
    RIGHT = 1
    LEFT  = 2
    UP    = 3
    DOWN  = 4


# Namedtuple untuk representasi titik di grid (x, y dalam pixel)
Point = namedtuple('Point', 'x, y')

# Urutan clockwise untuk konversi aksi relatif → absolut
# RIGHT → DOWN → LEFT → UP → RIGHT → ...
CLOCKWISE = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]


class SnakeGame:
    """
    Environment game Snake untuk training RL agent.

    Game ini dirancang untuk dua mode:
    1. Training mode (GAME_SPEED=0): tanpa delay, secepat mungkin
    2. Visualisasi mode (GAME_SPEED>0): dengan FPS tertentu untuk ditonton

    Attributes:
        w (int): Lebar window dalam pixel
        h (int): Tinggi window dalam pixel
        direction (Direction): Arah gerak ular saat ini
        head (Point): Posisi kepala ular
        snake (list[Point]): Seluruh body ular (head = index 0)
        food (Point): Posisi makanan saat ini
        score (int): Skor saat ini (jumlah makanan yang dimakan)
        frame_iteration (int): Jumlah langkah sejak episode dimulai
    """

    def __init__(self, render=True):
        """
        Inisialisasi game Snake.

        Args:
            render (bool): Jika True, tampilkan window Pygame.
                           Jika False, jalankan tanpa rendering (headless mode).
        """
        self.w = config.GRID_SIZE * config.BLOCK_SIZE  # 20 * 30 = 600px
        self.h = config.GRID_SIZE * config.BLOCK_SIZE  # 20 * 30 = 600px
        self.render_enabled = render

        if self.render_enabled:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake RL Agent')
            self.clock = pygame.time.Clock()

        # Inisialisasi state game pertama kali
        self.reset()

    def reset(self):
        """
        Reset game ke kondisi awal untuk memulai episode baru.

        Ular dimulai di tengah grid, menghadap ke kanan, dengan panjang 3 blok.
        Makanan ditempatkan secara random.

        Returns:
            np.ndarray: State vector 11 dimensi setelah reset
        """
        # Arah awal: ke kanan
        self.direction = Direction.RIGHT

        # Posisi awal kepala di tengah grid
        center_x = (config.GRID_SIZE // 2) * config.BLOCK_SIZE
        center_y = (config.GRID_SIZE // 2) * config.BLOCK_SIZE
        self.head = Point(center_x, center_y)

        # Body ular: kepala + 2 segmen di belakangnya (total panjang 3)
        self.snake = [
            self.head,
            Point(self.head.x - config.BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * config.BLOCK_SIZE), self.head.y)
        ]

        self.score = 0
        self.food = None
        self._place_food()  # Tempatkan makanan pertama
        self.frame_iteration = 0

        return self.get_state()

    def _place_food(self):
        """
        Tempatkan makanan di posisi random yang TIDAK ditempati oleh ular.

        Menggunakan loop untuk memastikan makanan tidak spawn di atas body ular.
        """
        while True:
            x = random.randint(0, config.GRID_SIZE - 1) * config.BLOCK_SIZE
            y = random.randint(0, config.GRID_SIZE - 1) * config.BLOCK_SIZE
            self.food = Point(x, y)
            # Pastikan makanan tidak spawn di atas ular
            if self.food not in self.snake:
                break

    def play_step(self, action):
        """
        Eksekusi satu langkah permainan berdasarkan aksi yang diberikan.

        Ini adalah interface utama yang dipanggil oleh agent setiap step.

        Args:
            action (list[int]): Aksi dalam format one-hot relatif:
                [1, 0, 0] = lurus (tidak belok)
                [0, 1, 0] = belok kanan (clockwise)
                [0, 0, 1] = belok kiri (counter-clockwise)

        Returns:
            tuple: (reward, done, score)
                - reward (int): Sinyal reward untuk step ini
                - done (bool): True jika episode selesai (ular mati)
                - score (int): Skor saat ini
        """
        self.frame_iteration += 1

        # 1. Handle event Pygame (agar window tidak freeze)
        if self.render_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        # 2. Gerakkan ular berdasarkan aksi relatif
        self._move(action)
        self.snake.insert(0, self.head)  # Tambah kepala baru di depan

        # 3. Cek apakah game over
        reward = config.REWARD_STEP  # Default: reward langkah biasa (0)
        done = False

        if self._is_collision():
            # Ular menabrak dinding atau ekornya sendiri
            done = True
            reward = config.REWARD_DIE
            return reward, done, self.score

        # Safety check: kalau ular terlalu lama tanpa makan, hentikan episode
        # Mencegah agent berputar-putar tanpa tujuan (infinite loop)
        # Batas = 100 langkah per panjang ular — semakin panjang, semakin banyak waktu
        if self.frame_iteration > 100 * len(self.snake):
            done = True
            reward = config.REWARD_DIE
            return reward, done, self.score

        # 4. Cek apakah ular makan makanan
        if self.head == self.food:
            self.score += 1
            reward = config.REWARD_EAT
            self._place_food()  # Spawn makanan baru
        else:
            # Kalau tidak makan, hapus ekor (ular tetap panjang yang sama)
            self.snake.pop()

        # 5. Update tampilan
        if self.render_enabled:
            self._update_ui()
            if config.GAME_SPEED > 0:
                self.clock.tick(config.GAME_SPEED)

        return reward, done, self.score

    def _move(self, action):
        """
        Update arah dan posisi kepala ular berdasarkan aksi relatif.

        Konversi aksi relatif (lurus/kanan/kiri) ke arah absolut (NESW)
        menggunakan urutan clockwise.

        Args:
            action (list[int]): Aksi one-hot [lurus, kanan, kiri]
        """
        # Cari index arah saat ini dalam urutan clockwise
        idx = CLOCKWISE.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            # Lurus — arah tidak berubah
            new_direction = CLOCKWISE[idx]
        elif np.array_equal(action, [0, 1, 0]):
            # Belok kanan — satu langkah clockwise
            new_direction = CLOCKWISE[(idx + 1) % 4]
        else:  # [0, 0, 1]
            # Belok kiri — satu langkah counter-clockwise
            new_direction = CLOCKWISE[(idx - 1) % 4]

        self.direction = new_direction

        # Hitung posisi kepala baru berdasarkan arah
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += config.BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= config.BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += config.BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= config.BLOCK_SIZE

        self.head = Point(x, y)

    def _is_collision(self, point=None):
        """
        Cek apakah suatu titik menyebabkan collision (tabrakan).

        Collision terjadi jika titik:
        1. Keluar dari batas grid (nabrak dinding), ATAU
        2. Mengenai body ular (nabrak ekor sendiri)

        Args:
            point (Point, optional): Titik yang dicek. Default = kepala ular.

        Returns:
            bool: True jika collision terdeteksi
        """
        if point is None:
            point = self.head

        # Cek nabrak dinding (keluar batas grid)
        if (point.x >= self.w or point.x < 0 or
                point.y >= self.h or point.y < 0):
            return True

        # Cek nabrak ekor sendiri (head ada di body, skip index 0 = head itu sendiri)
        if point in self.snake[1:]:
            return True

        return False

    def get_state(self):
        """
        Konversi kondisi game saat ini menjadi state vector 11 dimensi.

        State vector ini yang akan menjadi input ke neural network.

        Format:
            [0]  danger_straight  — bahaya jika lurus
            [1]  danger_right     — bahaya jika belok kanan
            [2]  danger_left      — bahaya jika belok kiri
            [3]  dir_left         — sedang bergerak ke kiri
            [4]  dir_right        — sedang bergerak ke kanan
            [5]  dir_up           — sedang bergerak ke atas
            [6]  dir_down         — sedang bergerak ke bawah
            [7]  food_left        — makanan di sebelah kiri
            [8]  food_right       — makanan di sebelah kanan
            [9]  food_up          — makanan di atas
            [10] food_down        — makanan di bawah

        Returns:
            np.ndarray: Array float32 berisi 11 nilai (0 atau 1)
        """
        head = self.head
        block = config.BLOCK_SIZE

        # Titik-titik di sekitar kepala (satu langkah ke setiap arah)
        point_l = Point(head.x - block, head.y)
        point_r = Point(head.x + block, head.y)
        point_u = Point(head.x, head.y - block)
        point_d = Point(head.x, head.y + block)

        # Boolean arah saat ini
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # --- Danger signals (3) ---
            # Bahaya lurus: cek collision di titik searah pergerakan
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d)),

            # Bahaya kanan (clockwise dari arah saat ini)
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_d)),

            # Bahaya kiri (counter-clockwise dari arah saat ini)
            (dir_d and self._is_collision(point_r)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d)),

            # --- Direction (4) ---
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # --- Food location relative to head (4) ---
            self.food.x < head.x,  # Makanan di kiri
            self.food.x > head.x,  # Makanan di kanan
            self.food.y < head.y,  # Makanan di atas (y kecil = atas di Pygame)
            self.food.y > head.y,  # Makanan di bawah
        ]

        return np.array(state, dtype=np.float32)

    def _update_ui(self):
        """
        Render tampilan game ke window Pygame.

        Menggambar:
        1. Background hitam
        2. Grid lines (subtle)
        3. Body ular (hijau dengan outline lebih gelap)
        4. Makanan (merah)
        5. Score text di pojok kiri atas
        """
        self.display.fill(BLACK)

        # Gambar grid lines untuk visualisasi yang lebih jelas
        for x in range(0, self.w, config.BLOCK_SIZE):
            pygame.draw.line(self.display, GRAY, (x, 0), (x, self.h))
        for y in range(0, self.h, config.BLOCK_SIZE):
            pygame.draw.line(self.display, GRAY, (0, y), (self.w, y))

        # Gambar ular — setiap segmen adalah kotak hijau dengan border lebih gelap
        for i, point in enumerate(self.snake):
            # Body utama (hijau)
            pygame.draw.rect(
                self.display, GREEN,
                pygame.Rect(point.x, point.y, config.BLOCK_SIZE, config.BLOCK_SIZE)
            )
            # Inner rect lebih kecil untuk efek 3D sederhana
            pygame.draw.rect(
                self.display, GREEN_DARK,
                pygame.Rect(point.x + 4, point.y + 4,
                            config.BLOCK_SIZE - 8, config.BLOCK_SIZE - 8)
            )

        # Gambar makanan (merah dengan highlight)
        pygame.draw.rect(
            self.display, RED,
            pygame.Rect(self.food.x, self.food.y,
                        config.BLOCK_SIZE, config.BLOCK_SIZE)
        )
        pygame.draw.rect(
            self.display, RED_LIGHT,
            pygame.Rect(self.food.x + 4, self.food.y + 4,
                        config.BLOCK_SIZE - 8, config.BLOCK_SIZE - 8)
        )

        # Tampilkan score di pojok kiri atas
        text = FONT.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, (5, 5))

        pygame.display.flip()


# =============================================================================
# Mode testing manual — jalankan file ini langsung untuk bermain dengan keyboard
# Berguna untuk memverifikasi bahwa game logic bekerja dengan benar
# Usage: python game/snake_game.py
# =============================================================================
if __name__ == '__main__':
    pygame.init()

    game = SnakeGame(render=True)

    # Override game speed untuk mode manual (lebih lambat agar bisa dimainkan)
    manual_fps = 10

    print("=== SNAKE GAME — Manual Testing Mode ===")
    print("Controls: Arrow keys / WASD")
    print("Tekan ESC atau tutup window untuk keluar")
    print("=========================================")

    while True:
        # Reset game untuk episode baru
        game.reset()
        done = False

        while not done:
            # Default: lurus
            action = [1, 0, 0]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

                    # Konversi input keyboard (absolut) ke aksi relatif
                    # berdasarkan arah saat ini
                    idx = CLOCKWISE.index(game.direction)

                    if event.key in (pygame.K_RIGHT, pygame.K_d):
                        target = Direction.RIGHT
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        target = Direction.LEFT
                    elif event.key in (pygame.K_UP, pygame.K_w):
                        target = Direction.UP
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        target = Direction.DOWN
                    else:
                        target = game.direction  # Key tidak dikenal, lurus

                    target_idx = CLOCKWISE.index(target)
                    diff = (target_idx - idx) % 4

                    if diff == 0:
                        action = [1, 0, 0]  # Lurus
                    elif diff == 1:
                        action = [0, 1, 0]  # Belok kanan
                    elif diff == 3:
                        action = [0, 0, 1]  # Belok kiri
                    else:
                        # diff == 2 berarti 180° (balik arah) — tidak diperbolehkan
                        # Ular akan terus lurus karena balik arah = bunuh diri
                        action = [1, 0, 0]

            reward, done, score = game.play_step(action)
            game.clock.tick(manual_fps)

        print(f"Game Over! Score: {score}")
        print("Memulai game baru dalam 2 detik...")
        pygame.time.wait(2000)
