import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
try:
    import cv2
    VISUALIZE = True
except ImportError:
    VISUALIZE = False
    print("OpenCV bulunamadı, görselleştirme devre dışı bırakıldı.")
import os
from collections import deque

# --- KONFİGÜRASYON ---
BATCH_SIZE = 512
LR = 1e-4 # Daha yavaş ama daha kararlı öğrenme
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995 
MEMORY_SIZE = 30000
MAX_EPISODES = 2000
BLOCK_SIZE = 30 
BOARD_HEIGHT = 20
BOARD_WIDTH = 10
RENDER_EVERY = 1  # Her oyunu göster
WAIT_TIME = 200   # Oyunu daha da yavaşlat (200ms)

# --- TETRIS OYUN MOTORU ---
class Tetris:
    def __init__(self):
        # Parçalar ve renkleri
        self.shapes = [
            [[1, 1, 1, 1]], # I
            [[1, 1], [1, 1]], # O
            [[1, 1, 0], [0, 1, 1]], # Z
            [[0, 1, 1], [1, 1, 0]], # S
            [[1, 1, 1], [0, 1, 0]], # T
            [[1, 1, 1], [1, 0, 0]], # L
            [[1, 1, 1], [0, 0, 1]]  # J
        ]
        self.colors = [
            (0, 255, 255), (0, 255, 0), (0, 0, 255), 
            (255, 0, 0), (255, 0, 255), (255, 165, 0), (0, 0, 128)
        ]
        
        self.high_score = 0
        self.episode = 0
        self.epsilon = 0.0
        self.wait_time = 100 # Varsayılan hız (ms)
        self.reset()
    
    # ... (Diğer metodlar aynı kalıyor) ...

    def step(self, action):
        # Action: (target_x, num_rotations)
        target_x, num_rotations = action
        
        # --- ANİMASYON BAŞLANGICI ---
        if VISUALIZE:
            # Mevcut parça özellikleri
            current_shape = self.current_piece['shape']
            current_id = self.current_piece['id']
            # Başlangıç pozisyonu (Genellikle ortada başlar ama burada basitleştirmek için 
            # animasyonu mantıksal x'e göre değil, görsel olarak kaydırarak yapacağız)
            
            # 1. Döndürme Animasyonu
            anim_shape = current_shape
            for _ in range(num_rotations):
                self.draw_animation_frame(anim_shape, 3, 0) # 3: Yaklaşık orta x, 0: Tepe y
                anim_shape = self._rotate(anim_shape)
            
            # Parçanın son hali
            final_shape = anim_shape
            shape_width = len(final_shape[0])
            target_x = max(0, min(BOARD_WIDTH - shape_width, target_x))
            
            # 2. Yatay Hareket Animasyonu
            # Başlangıç X'i ortadan (örn 3) alıp hedefe kaydıralım
            start_x = 3
            step_dir = 1 if target_x > start_x else -1
            if start_x != target_x:
                for x in range(start_x, target_x + step_dir, step_dir):
                    self.draw_animation_frame(final_shape, x, 0)
            
            # 3. Düşme Animasyonu
            drop_y = 0
            while not self._check_collision(self.board, final_shape, (target_x, drop_y + 1)):
                drop_y += 1
                # Her adımda değil, biraz hızlandırmak için 2 adımda bir veya hızlı çizim
                # Ama net görmek istiyorsanız her adımda çizelim:
                self.draw_animation_frame(final_shape, target_x, drop_y)

        # --- ANİMASYON BİTİŞİ ---
        
        # Mantıksal işlem (Eski kodun aynısı)
        shape = self.current_piece['shape']
        for _ in range(num_rotations):
            shape = self._rotate(shape)
        
        shape_width = len(shape[0])
        x = max(0, min(BOARD_WIDTH - shape_width, target_x))
        
        y = 0
        while not self._check_collision(self.board, shape, (x, y + 1)):
            y += 1
        
        self._place_piece(self.board, shape, (x, y))
        
        lines_cleared = self._clear_lines()
        
        if np.any(self.board[0]):
            self.game_over = True

        self.score += 1 + (lines_cleared ** 2) * 10
        self.cleared_lines += lines_cleared
        
        self.current_piece = self.next_piece
        self.next_piece = self._get_new_piece()
        
        if not self.game_over and self._check_collision(self.board, self.current_piece['shape'], (4, 0)):
            self.game_over = True
            
        # Ödül hesaplama (Aynı)
        _, board_stats = self._get_lines_holes_bumpiness(self.board)
        current_holes = board_stats['holes']
        current_bumpiness = board_stats['bumpiness']
        current_height = board_stats['height']
        
        reward = 0
        if lines_cleared > 0:
            reward += (lines_cleared ** 2) * 25
        reward -= (current_holes * 4)
        reward -= (current_bumpiness * 1)
        reward -= (current_height * 1)
        if self.game_over:
            reward -= 100
            
        return reward, self.game_over, self.score

    def draw_animation_frame(self, shape, x, y):
        # Bu fonksiyon render'a benzer ama geçici parçayı da çizer
        if not VISUALIZE: return
        
        # Geçici bir tahta kopyası oluştur ve üzerine parçayı çiz
        temp_board = self.board.copy()
        
        # Parçayı çiz
        piece_id = self.current_piece['id'] + 1
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell and 0 <= y + r < BOARD_HEIGHT and 0 <= x + c < BOARD_WIDTH:
                    temp_board[y + r][x + c] = piece_id # Geçici olarak parçayı ekle
                    
        # Render fonksiyonunu bu geçici tahta ile çağırabilirdik ama 
        # render self.board kullanıyor. O yüzden render kodunu buraya kopyalayıp özelleştirelim
        # Veya self.board'u geçici değiştirip geri alalım (Daha pratik)
        original_board = self.board
        self.board = temp_board
        self.render(wait_time=max(1, int(self.wait_time / 2))) # Animasyon biraz daha seri olsun
        self.board = original_board

    def render(self, wait_time=None):
        if not VISUALIZE:
            return
            
        if wait_time is None:
            wait_time = self.wait_time
            
        # Ekran boyutları (Oyun Alanı + Bilgi Paneli)
        panel_width = 200
        game_width = BOARD_WIDTH * BLOCK_SIZE
        height = BOARD_HEIGHT * BLOCK_SIZE
        
        img = np.zeros((height, game_width + panel_width, 3), dtype=np.uint8)
        
        # Arka plan
        img[:, game_width:] = (30, 30, 30)
        
        # Oyun Alanı Çizimi
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                val = self.board[y][x]
                if val > 0:
                    color = self.colors[(val - 1) % len(self.colors)]
                    cv2.rectangle(img, (x*BLOCK_SIZE, y*BLOCK_SIZE), 
                                  ((x+1)*BLOCK_SIZE, (y+1)*BLOCK_SIZE), color, -1)
                    cv2.rectangle(img, (x*BLOCK_SIZE, y*BLOCK_SIZE), 
                                  ((x+1)*BLOCK_SIZE, (y+1)*BLOCK_SIZE), (0, 0, 0), 1)
                else:
                    cv2.rectangle(img, (x*BLOCK_SIZE, y*BLOCK_SIZE), 
                                  ((x+1)*BLOCK_SIZE, (y+1)*BLOCK_SIZE), (20, 20, 20), 1)

        # Bilgi Paneli Metinleri
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 1
        x_start = game_width + 10
        y_start = 40
        line_height = 30
        
        cv2.putText(img, "TETRIS AI", (x_start, y_start), font, 0.8, (0, 255, 255), 2)
        cv2.putText(img, f"Episode: {self.episode}", (x_start, y_start + line_height * 2), font, font_scale, color, thickness)
        cv2.putText(img, f"High Score: {self.high_score}", (x_start, y_start + line_height * 3), font, font_scale, (0, 255, 0), thickness)
        cv2.putText(img, f"Score: {self.score}", (x_start, y_start + line_height * 4), font, font_scale, color, thickness)
        cv2.putText(img, f"Lines: {self.cleared_lines}", (x_start, y_start + line_height * 5), font, font_scale, color, thickness)
        cv2.putText(img, f"Epsilon: {self.epsilon:.3f}", (x_start, y_start + line_height * 7), font, font_scale, (200, 200, 200), thickness)
        
        # Hız Bilgisi
        cv2.putText(img, f"Speed: {self.wait_time}ms", (x_start, y_start + line_height * 8), font, font_scale, (100, 255, 100), thickness)
        
        # Sıradaki Parça
        cv2.putText(img, "Next:", (x_start, y_start + line_height * 10), font, font_scale, color, thickness)
        
        if hasattr(self, 'next_piece') and self.next_piece:
            next_shape = self.next_piece['shape']
            next_color = self.next_piece['color']
            preview_x = x_start
            preview_y = y_start + line_height * 11
            mini_block_size = 20
            
            for r, row in enumerate(next_shape):
                for c, cell in enumerate(row):
                    if cell:
                        cv2.rectangle(img, 
                                      (preview_x + c * mini_block_size, preview_y + r * mini_block_size), 
                                      (preview_x + (c + 1) * mini_block_size, preview_y + (r + 1) * mini_block_size), 
                                      next_color, -1)
                        cv2.rectangle(img, 
                                      (preview_x + c * mini_block_size, preview_y + r * mini_block_size), 
                                      (preview_x + (c + 1) * mini_block_size, preview_y + (r + 1) * mini_block_size), 
                                      (0, 0, 0), 1)

        cv2.line(img, (game_width, 0), (game_width, height), (100, 100, 100), 2)
        cv2.imshow("Tetris AI", img)
        
        # --- KLAVYE KONTROLÜ ---
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('w'): # Hızlandır (Süreyi azalt)
            self.wait_time = max(1, self.wait_time - 50)
        elif key == ord('s'): # Yavaşlat (Süreyi artır)
            self.wait_time = min(2000, self.wait_time + 50)
        elif key == ord('q'):
            exit()

    def reset(self):
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.score = 0
        self.cleared_lines = 0
        self.game_over = False
        self.bag = list(range(len(self.shapes)))
        random.shuffle(self.bag)
        self.current_piece = self._get_new_piece()
        self.next_piece = self._get_new_piece()
        return self._get_board_properties(self.board)

    def reset(self):
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        self.score = 0
        self.cleared_lines = 0
        self.game_over = False
        self.bag = list(range(len(self.shapes)))
        random.shuffle(self.bag)
        self.current_piece = self._get_new_piece()
        self.next_piece = self._get_new_piece()
        return self._get_board_properties(self.board)

    def _get_new_piece(self):
        if not self.bag:
            self.bag = list(range(len(self.shapes)))
            random.shuffle(self.bag)
        shape_idx = self.bag.pop()
        return {
            'shape': self.shapes[shape_idx],
            'color': self.colors[shape_idx],
            'id': shape_idx
        }

    def _rotate(self, shape):
        return [list(row) for row in zip(*shape[::-1])]

    def _check_collision(self, board, shape, offset):
        off_x, off_y = offset
        for cy, row in enumerate(shape):
            for cx, cell in enumerate(row):
                if cell:
                    if (cx + off_x < 0 or cx + off_x >= BOARD_WIDTH or
                            cy + off_y >= BOARD_HEIGHT or
                            (cy + off_y >= 0 and board[cy + off_y][cx + off_x])):
                        return True
        return False

    def step(self, action):
        # Action: (target_x, num_rotations)
        target_x, num_rotations = action
        
        # --- ANİMASYON BAŞLANGICI ---
        if VISUALIZE:
            # Mevcut parça özellikleri
            current_shape = self.current_piece['shape']
            
            # 1. Döndürme Animasyonu
            anim_shape = current_shape
            for _ in range(num_rotations):
                self.draw_animation_frame(anim_shape, 3, 0) # 3: Yaklaşık orta x, 0: Tepe y
                anim_shape = self._rotate(anim_shape)
            
            # Parçanın son hali
            final_shape = anim_shape
            shape_width = len(final_shape[0])
            target_x = max(0, min(BOARD_WIDTH - shape_width, target_x))
            
            # 2. Yatay Hareket Animasyonu
            # Başlangıç X'i ortadan (örn 3) alıp hedefe kaydıralım
            start_x = 3
            step_dir = 1 if target_x > start_x else -1
            if start_x != target_x:
                for x in range(start_x, target_x + step_dir, step_dir):
                    self.draw_animation_frame(final_shape, x, 0)
            
            # 3. Düşme Animasyonu
            drop_y = 0
            while not self._check_collision(self.board, final_shape, (target_x, drop_y + 1)):
                drop_y += 1
                # Her adımda çizelim:
                self.draw_animation_frame(final_shape, target_x, drop_y)

        # --- ANİMASYON BİTİŞİ ---
        
        shape = self.current_piece['shape']
        for _ in range(num_rotations):
            shape = self._rotate(shape)
        
        shape_width = len(shape[0])
        x = max(0, min(BOARD_WIDTH - shape_width, target_x))
        
        # Parçayı aşağı indir (Drop)
        y = 0
        while not self._check_collision(self.board, shape, (x, y + 1)):
            y += 1
        
        # Parçayı tahtaya yerleştir
        self._place_piece(self.board, shape, (x, y))
        
        # Satır silme kontrolü
        lines_cleared = self._clear_lines()
        
        # Eğer tahtanın en üst satırında blok varsa oyun biter (Top out)
        if np.any(self.board[0]):
            self.game_over = True

        self.score += 1 + (lines_cleared ** 2) * 10
        self.cleared_lines += lines_cleared
        
        # Yeni parça
        self.current_piece = self.next_piece
        self.next_piece = self._get_new_piece()
        
        # Oyun bitti mi?
        if not self.game_over and self._check_collision(self.board, self.current_piece['shape'], (4, 0)):
            self.game_over = True
            
        # --- GELİŞMİŞ ÖDÜL SİSTEMİ (Reward Shaping) ---
        # Mevcut tahtanın durumunu analiz et
        _, board_stats = self._get_lines_holes_bumpiness(self.board)
        current_holes = board_stats['holes']
        current_bumpiness = board_stats['bumpiness']
        current_height = board_stats['height']
        
        reward = 0
        
        # 1. Satır Silme Ödülü (Karesel artış - Tetris yapmak çok değerli)
        if lines_cleared > 0:
            reward += (lines_cleared ** 2) * 25
            
        # 2. Durum Cezaları (Tahtayı kötü kullandıysa ceza ver)
        reward -= (current_holes * 4)       # Her boşluk için -4 puan
        reward -= (current_bumpiness * 1)   # Her pürüzlülük için -1 puan
        reward -= (current_height * 1)      # Toplam yükseklik için -1 puan
        
        # 3. Oyun Sonu Cezası
        if self.game_over:
            reward -= 100 # Kaybetmek çok kötü
            
        return reward, self.game_over, self.score

    def draw_animation_frame(self, shape, x, y):
        # Bu fonksiyon render'a benzer ama geçici parçayı da çizer
        if not VISUALIZE: return
        
        # Geçici bir tahta kopyası oluştur ve üzerine parçayı çiz
        temp_board = self.board.copy()
        
        # Parçayı çiz
        piece_id = self.current_piece['id'] + 1
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell and 0 <= y + r < BOARD_HEIGHT and 0 <= x + c < BOARD_WIDTH:
                    temp_board[y + r][x + c] = piece_id # Geçici olarak parçayı ekle
                    
        # Render fonksiyonunu bu geçici tahta ile çağırabilirdik ama 
        # render self.board kullanıyor. O yüzden render kodunu buraya kopyalayıp özelleştirelim
        # Veya self.board'u geçici değiştirip geri alalım (Daha pratik)
        original_board = self.board
        self.board = temp_board
        self.render(wait_time=max(1, int(self.wait_time / 2))) # Animasyon biraz daha seri olsun
        self.board = original_board

    def _place_piece(self, board, shape, offset):
        # Renk bilgisini almak için mevcut parçanın ID'sini kullanıyoruz
        # Simülasyon sırasında (get_next_states) ID'yi bilmiyorsak varsayılan 1 kullanırız
        # Ancak ana oyun döngüsünde self.current_piece['id'] erişilebilir.
        
        piece_id = 1
        if hasattr(self, 'current_piece') and self.current_piece:
             piece_id = self.current_piece['id'] + 1
             
        off_x, off_y = offset
        for cy, row in enumerate(shape):
            for cx, cell in enumerate(row):
                if cell and 0 <= cy + off_y < BOARD_HEIGHT and 0 <= cx + off_x < BOARD_WIDTH:
                    board[cy + off_y][cx + off_x] = piece_id

    def _clear_lines(self):
        new_board = [row for row in self.board if not all(row)]
        lines_cleared = BOARD_HEIGHT - len(new_board)
        for _ in range(lines_cleared):
            new_board.insert(0, [0] * BOARD_WIDTH)
        self.board = np.array(new_board)
        return lines_cleared

    # --- ÖZELLİK ÇIKARIMI (FEATURE EXTRACTION) ---
    # Yapay zekanın tüm tahtayı piksel piksel öğrenmesi zordur.
    # Bunun yerine ona tahtanın "durumu" hakkında özet bilgiler veriyoruz.
    def _get_board_properties(self, board):
        lines, rows = self._get_lines_holes_bumpiness(board)
        # Normalizasyonu kaldırdık, ham verilerle çalışacak
        return torch.tensor([
            lines, 
            rows['holes'], 
            rows['bumpiness'], 
            rows['height']
        ], dtype=torch.float32)

    def _get_lines_holes_bumpiness(self, board):
        # Bu fonksiyon tahtadaki boşlukları, pürüzlülüğü ve yüksekliği hesaplar
        board = np.array(board)
        mask = board != 0
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), BOARD_HEIGHT)
        heights = BOARD_HEIGHT - invert_heights
        
        total_height = np.sum(heights)
        max_height = np.max(heights)
        
        diffs = np.abs(heights[:-1] - heights[1:])
        bumpiness = np.sum(diffs)
        
        holes = 0
        for i in range(BOARD_WIDTH):
            col = board[:, i]
            if np.any(col):
                # İlk dolu hücreden sonrasındaki boş hücreleri say
                first_block = np.argmax(col)
                holes += np.sum(col[first_block:] == 0)
                
        return 0, {'holes': holes, 'bumpiness': bumpiness, 'height': total_height}

    # Bir sonraki olası durumları simüle et
    def get_next_states(self):
        return self._get_possible_states(self.board, self.current_piece)

    def _get_possible_states(self, board, piece_dict):
        states = {}
        piece = piece_dict['shape']
        piece_id = piece_dict['id']
        
        # Parçanın tüm rotasyonları için (0, 90, 180, 270)
        num_rotations = 4
        if piece_id == 1: num_rotations = 1 # O parçası dönmez
        elif piece_id in [0, 2, 3]: num_rotations = 2 # I, S, Z iki durumda aynıdır
        
        for rot in range(num_rotations):
            rotated_piece = piece
            for _ in range(rot):
                rotated_piece = self._rotate(rotated_piece)
            
            width = len(rotated_piece[0])
            
            # Her sütun için dene
            for x in range(BOARD_WIDTH - width + 1):
                board_copy = board.copy()
                
                # Drop simülasyonu
                y = 0
                while not self._check_collision(board_copy, rotated_piece, (x, y + 1)):
                    y += 1
                
                self._place_piece(board_copy, rotated_piece, (x, y))
                
                # Özellikleri çıkar
                lines_cleared = np.sum(np.all(board_copy, axis=1))
                board_props = self._get_lines_holes_bumpiness(board_copy)
                
                # Giriş verisi: [Silinen Satır, Boşluklar, Pürüzlülük, Toplam Yükseklik]
                features = torch.tensor([
                    lines_cleared, 
                    board_props[1]['holes'], 
                    board_props[1]['bumpiness'], 
                    board_props[1]['height']
                ], dtype=torch.float32)
                
                # Dönüş: {Hamle: (Özellikler, Yeni_Tahta_Durumu)}
                states[(x, rot)] = (features, board_copy)
                
        return states

    def render(self, wait_time=1):
        if not VISUALIZE:
            return
            
        # Ekran boyutları (Oyun Alanı + Bilgi Paneli)
        panel_width = 200
        game_width = BOARD_WIDTH * BLOCK_SIZE
        height = BOARD_HEIGHT * BLOCK_SIZE
        
        img = np.zeros((height, game_width + panel_width, 3), dtype=np.uint8)
        
        # Arka plan (Panel için koyu gri)
        img[:, game_width:] = (30, 30, 30)
        
        # Oyun Alanı Çizimi
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                val = self.board[y][x]
                if val > 0:
                    color = self.colors[(val - 1) % len(self.colors)]
                    cv2.rectangle(img, (x*BLOCK_SIZE, y*BLOCK_SIZE), 
                                  ((x+1)*BLOCK_SIZE, (y+1)*BLOCK_SIZE), color, -1)
                    cv2.rectangle(img, (x*BLOCK_SIZE, y*BLOCK_SIZE), 
                                  ((x+1)*BLOCK_SIZE, (y+1)*BLOCK_SIZE), (0, 0, 0), 1)
                else:
                    # Boş alan ızgarası
                    cv2.rectangle(img, (x*BLOCK_SIZE, y*BLOCK_SIZE), 
                                  ((x+1)*BLOCK_SIZE, (y+1)*BLOCK_SIZE), (20, 20, 20), 1)

        # Bilgi Paneli Metinleri
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 1
        x_start = game_width + 10
        y_start = 40
        line_height = 30
        
        cv2.putText(img, "TETRIS AI", (x_start, y_start), font, 0.8, (0, 255, 255), 2)
        
        cv2.putText(img, f"Episode: {self.episode}", (x_start, y_start + line_height * 2), font, font_scale, color, thickness)
        cv2.putText(img, f"High Score: {self.high_score}", (x_start, y_start + line_height * 3), font, font_scale, (0, 255, 0), thickness)
        cv2.putText(img, f"Score: {self.score}", (x_start, y_start + line_height * 4), font, font_scale, color, thickness)
        cv2.putText(img, f"Lines: {self.cleared_lines}", (x_start, y_start + line_height * 5), font, font_scale, color, thickness)
        cv2.putText(img, f"Epsilon: {self.epsilon:.3f}", (x_start, y_start + line_height * 7), font, font_scale, (200, 200, 200), thickness)
        
        # --- SIRADAKİ PARÇA (NEXT PIECE) ---
        cv2.putText(img, "Next:", (x_start, y_start + line_height * 9), font, font_scale, color, thickness)
        
        next_shape = self.next_piece['shape']
        next_color = self.next_piece['color']
        
        # Önizleme kutusu başlangıç koordinatları
        preview_x = x_start
        preview_y = y_start + line_height * 10
        mini_block_size = 20 # Önizleme için daha küçük bloklar
        
        for r, row in enumerate(next_shape):
            for c, cell in enumerate(row):
                if cell:
                    cv2.rectangle(img, 
                                  (preview_x + c * mini_block_size, preview_y + r * mini_block_size), 
                                  (preview_x + (c + 1) * mini_block_size, preview_y + (r + 1) * mini_block_size), 
                                  next_color, -1)
                    cv2.rectangle(img, 
                                  (preview_x + c * mini_block_size, preview_y + r * mini_block_size), 
                                  (preview_x + (c + 1) * mini_block_size, preview_y + (r + 1) * mini_block_size), 
                                  (0, 0, 0), 1)

        # Ayırıcı çizgi
        cv2.line(img, (game_width, 0), (game_width, height), (100, 100, 100), 2)
        
        cv2.imshow("Tetris AI", img)
        cv2.waitKey(wait_time)

# --- YAPAY ZEKA MODELİ (DQN) ---
class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        # Giriş: 4 özellik
        self.fc1 = nn.Sequential(nn.Linear(4, 512), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True))
        self.fc3 = nn.Linear(512, 1) # Çıkış: Puan

        # Ağırlıkları başlatma (Heuristic'e yakın başlamak için)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return self.fc3(out)

# --- EĞİTİM ---
def train():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps" 
    else:
        device = "cpu"
        
    print(f"Device: {device}")
    
    env = Tetris()
    model = DeepQNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    replay_memory = deque(maxlen=MEMORY_SIZE)
    epsilon = EPSILON_START
    
    # Var olan modeli yükle
    if os.path.exists("tetris_dqn.pth"):
        print("Eğitilmiş model bulundu, yükleniyor...")
        try:
            model.load_state_dict(torch.load("tetris_dqn.pth", map_location=device))
            epsilon = 0.05 
            print(f"Model yüklendi. Epsilon: {epsilon}")
        except:
            print("Model dosyası uyumsuz, sıfırdan başlanıyor.")
    else:
        print("Kayıtlı model bulunamadı, sıfırdan başlanıyor.")
    
    print("Eğitim başlıyor... (Çıkmak için Ctrl+C)")
    
    for episode in range(MAX_EPISODES):
        env.reset()
        env.episode = episode + 1
        env.epsilon = epsilon
        state = env._get_board_properties(env.board) # Başlangıç durumu
        
        steps = 0
        total_reward = 0
        
        while not env.game_over:
            # 1. Mevcut parça için olası tüm hamleleri (sonraki durumları) al
            # next_states -> {action: (features, board)}
            next_states = env.get_next_states()
            
            if not next_states:
                break
            
            # 2. Epsilon-Greedy ile hamle seç
            if random.random() < epsilon:
                # RASTGELE SEÇİM
                action = random.choice(list(next_states.keys()))
                next_states_tensor_for_memory = next_states[action][0]
            else:
                # AKILLI SEÇİM (2-Adımlı Bakış)
                model.eval()
                
                best_action = None
                best_score = -float('inf')
                
                # Her olası birinci hamle için...
                for action1, (feat1, board1) in next_states.items():
                    # Bu hamleyi yapsak, sıradaki parça (next_piece) ile ne yapabiliriz?
                    step2_states = env._get_possible_states(board1, env.next_piece)
                    
                    if not step2_states:
                        # Eğer bu hamleden sonra sıradaki parça koyulamıyorsa (Oyun Bitiyor) -> Çok Kötü
                        score = -1000.0
                    else:
                        # Sıradaki parça için tüm olası durumların özelliklerini al
                        feats2 = [val[0] for val in step2_states.values()]
                        if not feats2:
                            score = -1000.0
                        else:
                            # Hepsini tek seferde modele sor (Batch Inference)
                            feats2_tensor = torch.stack(feats2).to(device)
                            with torch.no_grad():
                                scores = model(feats2_tensor)
                                # Gelecekteki en iyi durumu hedefle
                                score = torch.max(scores).item()
                    
                    if score > best_score:
                        best_score = score
                        best_action = action1
                
                action = best_action
                if action is None: # Nadir durumlar için güvenlik
                     action = random.choice(list(next_states.keys()))
                     
                next_states_tensor_for_memory = next_states[action][0]
                model.train()
            
            # 3. Hamleyi uygula
            reward, done, score = env.step(action)
            total_reward += reward
            
            # 4. Hafızaya kaydet
            replay_memory.append((next_states_tensor_for_memory, reward, done))
            
            # 5. Model Eğitimi (Replay Buffer)
            if len(replay_memory) > BATCH_SIZE:
                batch = random.sample(replay_memory, BATCH_SIZE)
                
                state_batch = torch.stack([x[0] for x in batch]).to(device)
                reward_batch = torch.tensor([x[1] for x in batch], dtype=torch.float32).to(device)
                done_batch = torch.tensor([x[2] for x in batch], dtype=torch.bool).to(device)
                
                q_values = model(state_batch)
                
                # Hedef Q Değeri: Reward + Gamma * Max(Next_Q)
                # Basit Tetris için genellikle sadece anlık ödülü maksimize etmek bile iyi sonuç verir
                # Ancak burada standart Q-Learning update'i yapıyoruz
                with torch.no_grad():
                    # Basit versiyonda bir sonraki adımı tahmin etmek yerine
                    # doğrudan iyi hamleleri (yüksek ödül) ödüllendiriyoruz.
                    target = reward_batch.unsqueeze(1)
                
                loss = criterion(q_values, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            steps += 1
            
            # Görselleştirme
            if episode % RENDER_EVERY == 0:
                env.render(wait_time=WAIT_TIME)
                
        # High Score Güncellemesi
        if env.score > env.high_score:
            env.high_score = env.score
                
        # Epsilon azaltma
        if epsilon > EPSILON_END:
            epsilon *= EPSILON_DECAY
            
        print(f"Episode: {episode+1}/{MAX_EPISODES}, Score: {env.score}, Lines: {env.cleared_lines}, Epsilon: {epsilon:.3f}")
        
        # Modeli kaydet
        if (episode + 1) % 25 == 0:
            torch.save(model.state_dict(), "tetris_dqn.pth")

    print("Eğitim tamamlandı.")
    if VISUALIZE:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    train()
