import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math

# ==============================================================================
# 1. KNOWLEDGE BASE
# ==============================================================================
class KnowledgeBase:
    def __init__(self):
        self.classes = ["Astronaut", "Diver", "Firefighter", "Hazmat"]
        
        self.raw_definitions = {
            "Astronaut":   ["Suit", "Helmet", "Gloves", "Oxygen Tank", "Tether", "Heat Shield", "Zero-G"],
            "Diver":       ["Suit", "Helmet", "Gloves", "Oxygen Tank", "Tether", "Rubber Seal", "Deep Water"],
            "Firefighter": ["Suit", "Helmet", "Gloves", "Oxygen Tank", "Heat Shield", "Smoke/Fire"],
            "Hazmat":      ["Suit", "Helmet", "Gloves", "Rubber Seal", "Biohazard"]
        }
        
        self.all_features = sorted(list(set(f for t in self.raw_definitions.values() for f in t)))
        self.dim = len(self.all_features)
        
        # --- IDF WEIGHTING ---
        self.feature_weights = {}
        for f in self.all_features:
            count = sum(1 for t in self.raw_definitions.values() if f in t)
            # Log weighting: Rare = Heavy
            weight = math.log(8.0 / count) 
            self.feature_weights[f] = weight

        # --- CALCULATE MEMORY MASS (The Capacity) ---
        # We need to know how much energy a "Perfect Match" contains.
        self.memory_masses = []
        for name in self.classes:
            mass = sum(self.feature_weights[f] for f in self.raw_definitions[name])
            self.memory_masses.append(mass)
        
        # The "Standard Candle" for Unknown: Average Mass of a Class
        self.avg_mass = sum(self.memory_masses) / len(self.memory_masses)
        # print(f"Average Memory Mass: {self.avg_mass:.2f}")

    def get_memory_matrix(self):
        matrix = []
        for name in self.classes:
            vec = torch.full((self.dim,), -1.0)
            for f in self.raw_definitions[name]:
                vec[self.all_features.index(f)] = 1.0
            matrix.append(vec)
        return torch.stack(matrix)

    def get_clue_vector(self, clue_name):
        vec = torch.zeros(self.dim)
        if clue_name in self.all_features:
            idx = self.all_features.index(clue_name)
            vec[idx] = self.feature_weights[clue_name]
        return vec

# ==============================================================================
# 2. NEURAL PHYSICS (Dynamic Null)
# ==============================================================================
class NeuralPhysics(torch.nn.Module):
    def __init__(self, memory_matrix, avg_mass, beta=2.0):
        super().__init__()
        self.memory = torch.nn.Parameter(memory_matrix, requires_grad=False)
        self.avg_mass = avg_mass # The Gravity of the Void
        self.beta = beta

    def relax(self, current_state, step_size=0.01):
        # 1. Similarity
        memory_norm = F.normalize(self.memory, dim=-1)
        similarity = torch.mm(current_state, memory_norm.T) # [1, 4]
        
        # 2. CALCULATE DYNAMIC NULL (Unrealized Potential)
        # How much evidence do we have?
        current_mass = torch.norm(current_state).item()
        
        # The Null Score is the gap between "Perfect Knowledge" and "Current Knowledge"
        # We clamp it at 0 so it doesn't become negative (which would mean we know MORE than everything)
        null_score = max(0.0, self.avg_mass - current_mass)
        
        # Append Null to logits
        null_tensor = torch.tensor([[null_score]])
        logits = torch.cat([similarity, null_tensor], dim=1) # [1, 5]
        
        # 3. Attention (Softmax over 5 options: 4 Classes + Unknown)
        attention = F.softmax(self.beta * logits, dim=-1)
        
        # Extract Unknown Probability
        unknown_prob = attention[0, 4].item()
        class_probs = attention[0, 0:4] # Rescale these? No, keep as is.
        
        # 4. Reconstruction (Only using the 4 classes)
        # We assume the "Unknown" contributes nothing to the dream image
        target_state = torch.mm(class_probs.unsqueeze(0), self.memory)
        
        # 5. Update
        new_state = (1 - step_size) * current_state + (step_size * target_state)
        
        return new_state, class_probs, unknown_prob

# ==============================================================================
# 3. VISUALIZER (Updated for Explicit Unknown)
# ==============================================================================
class Visualizer:
    def __init__(self, kb):
        self.kb = kb
        self.fig = None
        self.ax_radar = None
        self.human_guess = None

    def setup(self):
        self.human_guess = None
        plt.close('all')
        plt.ion()
        
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("The Semantic Race")
        self.fig.patch.set_facecolor('white')
        
        self.ax_radar = self.fig.add_subplot(111, polar=True)
        
        num_vars = len(self.kb.classes)
        self.angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        self.angles += self.angles[:1]
        
        self.radar_poly, = self.ax_radar.plot([], [], color='teal', linewidth=2)
        self.radar_fill = self.ax_radar.fill([], [], color='teal', alpha=0.25)
        
        # The UNKNOWN Bar (Center Orb logic replaced by explicit visual)
        self.orb = self.ax_radar.scatter([0], [0], s=100, color='gray', alpha=0.95, zorder=10)
        self.txt_unknown = self.fig.text(0.5, 0.5, "UNK", ha='center', va='center', color='white', weight='bold')
        
        labels_with_keys = [f"{cls}\n({i+1})" for i, cls in enumerate(self.kb.classes)]
        self.ax_radar.set_xticks(self.angles[:-1])
        self.ax_radar.set_xticklabels(labels_with_keys, size=11, weight='bold')
        self.ax_radar.set_ylim(0, 1.0)
        self.ax_radar.set_yticks([]) 
        
        self.txt_status = self.fig.text(0.5, 0.05, "Initializing...", ha='center', fontsize=14)
        self.txt_clues = self.fig.text(0.02, 0.98, "", va='top', fontsize=11, family='monospace', color='blue')
        
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

    def _on_key(self, event):
        if event.key in ['1', '2', '3', '4']:
            self.human_guess = int(event.key) - 1

    def update(self, probs, unknown_prob, status_msg, clues_msg):
        # FIX: Convert list to numpy array so we can divide it
        probs = np.array(probs)
        
        # Normalize display probabilities so they look good on radar
        # (Even if Unknown is 90%, we want to see the shape of the remaining 10%)
        disp_probs = probs / (np.sum(probs) + 1e-9)
        
        # Convert back to list for plotting concatenation
        values = disp_probs.tolist() + disp_probs.tolist()[:1]
        
        self.radar_poly.set_data(self.angles, values)
        if len(self.ax_radar.collections) > 1: self.ax_radar.collections[-1].remove()
        self.ax_radar.fill(self.angles, values, color='teal', alpha=0.25)
        
        # UNKNOWN ORB SIZE
        # Size scales with Unknown Probability
        orb_size = max(0, unknown_prob * 15000) 
        self.orb.set_sizes([orb_size])
        
        # Orb Color (Red if dangerous)
        # We check the max of the RAW probs (not display probs) for the red alert
        self.orb.set_color('red' if np.max(probs) > 0.8 else 'gray')

        self.txt_status.set_text(status_msg)
        self.txt_clues.set_text(clues_msg)
        plt.pause(0.02)
        
    def show_result(self, winner_type, target_name):
        color = '#d4ffd4' if winner_type == 'HUMAN' else '#ffd4d4'
        self.fig.patch.set_facecolor(color)
        self.ax_radar.set_facecolor(color)
        self.fig.text(0.5, 0.5, winner_type, ha='center', va='center', 
                      fontsize=40, weight='bold', color='black',
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        self.fig.text(0.5, 0.4, f"Identity: {target_name}", ha='center', fontsize=20)
        plt.draw()
        plt.pause(0.1)
        
    def close(self):
        plt.close(self.fig)

# ==============================================================================
# 4. GAME ENGINE
# ==============================================================================
class GameEngine:
    def __init__(self):
        self.kb = KnowledgeBase()
        # Pass avg_mass to Physics
        self.physics = NeuralPhysics(self.kb.get_memory_matrix(), self.kb.avg_mass, beta=2.0)
        self.ui = Visualizer(self.kb)
        
    def play_round(self):
        self.ui.setup()
        
        target_idx = random.randint(0, 3)
        target_name = self.kb.classes[target_idx]
        target_traits = self.kb.raw_definitions[target_name]
        
        # --- DECK LOGIC ---
        # Sort by weight (Common -> Rare)
        sorted_clues = sorted(target_traits, key=lambda x: self.kb.feature_weights[x])
        
        print(f"\n>>> NEW GAME. Target Hidden. Press 1-4 to guess.")
        
        current_state = torch.zeros((1, self.kb.dim))
        revealed_list = []
        game_over = False
        winner = None
        
        for i, clue in enumerate(sorted_clues):
            if game_over: break
            
            revealed_list.append(clue)
            clues_str = "\n".join(revealed_list[-8:]) 
            print(f"Clue {i+1}: {clue}")
            
            # 1. READ PHASE
            start_read = time.time()
            while (time.time() - start_read) < 2.0:
                # Peek at physics (step_size=0)
                _, attn, unk_prob = self.physics.relax(current_state, step_size=0.0)
                # Pass normalized probs for rendering
                self.ui.update(attn.squeeze().tolist(), unk_prob, "Reading...", clues_str)
                if self.ui.human_guess is not None:
                    game_over = True; winner = self.check_human(target_idx); break
            
            if game_over: break

            # 2. INJECT
            current_state = current_state + self.kb.get_clue_vector(clue)
            
            # 3. DRIFT PHASE
            start_time = time.time()
            duration = 5.0
            
            while (time.time() - start_time) < duration:
                # Physics Relax
                new_state, attn, unk_prob = self.physics.relax(current_state, step_size=0.01)
                probs = attn.squeeze().tolist()
                current_state = new_state 
                
                elapsed = time.time() - start_time
                self.ui.update(probs, unk_prob, f"Analyzing... {elapsed:.1f}s", clues_str)
                
                if self.ui.human_guess is not None:
                    game_over = True; winner = self.check_human(target_idx); break
                
                # --- NEW CONFIDENCE CHECK ---
                # We check the Raw Probability of the Best Class vs Unknown
                # We do NOT use the normalized race bars for this check.
                # We check if the AI is "Certain enough" (Unknown < 20%)
                
                best_class_prob = max(probs)
                
                # If the AI is more sure of a Class than it is Unknown...
                # AND it has a decent lead...
                if best_class_prob > 0.8 and unk_prob < 0.2:
                    ai_idx = probs.index(best_class_prob)
                    print(f">>> AI BUZZED IN: {self.kb.classes[ai_idx]}")
                    if ai_idx == target_idx: winner = "MACHINE"
                    else: winner = "HUMAN"
                    game_over = True
                    break
            
            if game_over: break
            
        if not winner: winner = "MACHINE"
        self.ui.show_result(winner, target_name)
        plt.pause(4.0)
        self.ui.close()

    def check_human(self, target_idx):
        guess = self.ui.human_guess
        guess_name = self.kb.classes[guess]
        print(f">>> HUMAN GUESS: {guess_name}")
        return "HUMAN" if guess == target_idx else "MACHINE"

if __name__ == "__main__":
    game = GameEngine()
    while True:
        game.play_round()
        if input("Again? (y/n): ").lower() != 'y': break