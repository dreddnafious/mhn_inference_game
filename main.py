import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import sys

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# The dimension of the semantic vector space. 
# We use a small dimension here for interpretability, but this scales to 4096+.
DIM_SIZE = 12

# Physics Parameters
BETA = 5.0          # Inverse Temperature. Controls how "sharp" the decision is.
STEP_SIZE = 0.05    # Viscosity. Lower = Slower "Drift" (More time for human to react).
THRESHOLD = 0.95    # Confidence level required for AI to "Buzz In".

# ==============================================================================
# MODULE 1: KNOWLEDGE BASE (The Prism / Embedding Layer)
# ==============================================================================
class KnowledgeBase:
    """
    Responsibilities:
    1. Define the 'Universe' (Classes and Features).
    2. Act as the EMBEDDING LAYER (Mapping Strings -> Vectors).
    3. Define the 'Truth' (The Ideal Memory Vectors).
    """
    def __init__(self):
        self.classes = ["Assassin", "Hunter", "Soldier", "Photographer"]
        
        # The Features (The Vocabulary of Clues)
        self.features = [
            "Shoot", "Scope", "Stealth", "Wait",   # Action-based
            "Trophy", "Target", "Orders", "Flash", # Object-based
            "Tripod", "Camouflage", "Bag", "Digital" # Gear-based
        ]
        
        # MATRIX DEFINITION
        # This is the "Prism". It maps Classes to Features.
        # 1.0 = Strong Association
        # 0.5 = Weak/Ambiguous Association
        # -1.0 = Hard Incompatibility (Repulsor)
        # 0.0 = Irrelevant
        
        # We define this manually to ensure High Ambiguity.
        self.definitions = {
            "Assassin": {
                "Shoot": 1.0, "Scope": 1.0, "Stealth": 1.0, "Wait": 1.0,
                "Target": 1.0, "Orders": 1.0, "Bag": 0.5,
                "Flash": -1.0, "Trophy": -1.0, "Tripod": -0.5 # Hates attention
            },
            "Hunter": {
                "Shoot": 1.0, "Scope": 1.0, "Stealth": 1.0, "Wait": 1.0,
                "Trophy": 1.0, "Camouflage": 1.0,
                "Orders": -1.0, "Digital": -0.5
            },
            "Soldier": {
                "Shoot": 1.0, "Scope": 0.5, "Orders": 1.0, "Camouflage": 1.0,
                "Bag": 1.0, "Stealth": 0.2,
                "Trophy": -0.5
            },
            "Photographer": {
                "Shoot": 0.8, # "Shoots" photos (Polysemy)
                "Scope": 0.5, # Telephoto lens
                "Wait": 1.0, "Flash": 1.0, "Tripod": 1.0, "Digital": 1.0, "Bag": 1.0,
                "Stealth": 0.2, # Wildlife photography requires stealth
                "Target": 0.5,  # Subject
                "Orders": 0.5   # Client orders
            }
        }

    def get_memory_matrix(self):
        """
        Converts the definitions into a Dense Tensor for the Hopfield Network.
        Returns: Tensor [Num_Classes, Dim_Size]
        """
        matrix = []
        for class_name in self.classes:
            vec = torch.zeros(DIM_SIZE)
            def_dict = self.definitions[class_name]
            
            for i, feature in enumerate(self.features):
                # If defined, use weight. Else 0.0.
                vec[i] = def_dict.get(feature, 0.0)
            matrix.append(vec)
        
        return torch.stack(matrix)

    def get_clue_vector(self, clue_name):
        """
        This is the PROJECTION step.
        When a clue is revealed, we don't just flip a bit.
        We project it into the latent space based on its semantic associations.
        
        For this game, we simplify: The clue vector is a One-Hot vector 
        at the feature's index. The 'Meaning' comes from how it interacts 
        with the Memory Matrix during the physics step.
        """
        vec = torch.zeros(DIM_SIZE)
        if clue_name in self.features:
            idx = self.features.index(clue_name)
            vec[idx] = 1.0
        return vec

# ==============================================================================
# MODULE 2: NEURAL PHYSICS (The Brain / Reusable Core)
# ==============================================================================
class NeuralPhysics(torch.nn.Module):
    """
    Responsibilities:
    1. Store memories (Attractors).
    2. Perform Energy Relaxation (Thinking).
    3. Pure math. No game logic.
    """
    def __init__(self, memory_matrix, beta=BETA):
        super().__init__()
        # We store memories as a fixed parameter. 
        # requires_grad=False because we are doing Inference, not Backprop.
        self.memory = torch.nn.Parameter(memory_matrix, requires_grad=False)
        self.beta = beta

    def relax(self, current_state, step_size=STEP_SIZE):
        """
        Performs one step of Hopfield dynamics.
        
        Args:
            current_state: The thought vector [1, Dim]
            step_size: The 'Viscosity'. 
                       0.0 = Frozen. 
                       1.0 = Teleport to solution.
                       0.05 = Slow drift (Good for visualization).
        """
        # 1. Similarity Calculation (State @ Memory.T)
        # Note: We do NOT normalize the State here. 
        # This allows the magnitude of input (evidence accumulation) to matter.
        memory_norm = F.normalize(self.memory, dim=-1)
        similarity = torch.mm(current_state, memory_norm.T)
        
        # 2. Attention Mechanism (The Energy Function)
        # Softmax converts raw similarity into probability distribution.
        attention = F.softmax(self.beta * similarity, dim=-1)
        
        # 3. Reconstruction (The Dream)
        # What should the state look like based on these probabilities?
        target_state = torch.mm(attention, self.memory)
        
        # 4. The Update (Euler Integration)
        # New = Old + (Target - Old) * Rate
        new_state = (1 - step_size) * current_state + (step_size * target_state)
        
        return new_state, attention


# ==============================================================================
# MODULE 3: VISUALIZER (Fixed for Interactivity)
# ==============================================================================
import matplotlib.gridspec as gridspec

class Visualizer:
    def __init__(self, kb):
        self.kb = kb
        self.fig = None
        self.ax_heat = None
        self.ax_race = None
        self.ax_text = None
        self.human_guess = None 

    def setup(self):
        self.human_guess = None 
        plt.close('all') 
        plt.ion() 
        
        self.fig = plt.figure(figsize=(16, 6))
        self.fig.canvas.manager.set_window_title("The Semantic Race")
        
        # Reset Background Color (in case it was red/green from last game)
        self.fig.patch.set_facecolor('white')

        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 2, 1.5])

        # Connect Keyboard
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # AXIS 1: Heatmap (Left - Smaller)
        self.ax_heat = self.fig.add_subplot(gs[0])
        self.img_dream = self.ax_heat.imshow(np.zeros((1, DIM_SIZE)), cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        self.ax_heat.set_title("Brain Scan", fontsize=10)
        self.ax_heat.set_yticks([])
        # Vertical presentation for compactness
        self.ax_heat.set_xticks(range(DIM_SIZE))
        self.ax_heat.set_xticklabels(self.kb.features, rotation=90, fontsize=8)

        # AXIS 2: Confidence Race (Middle - Main Focus)
        self.ax_race = self.fig.add_subplot(gs[1])
        self.bars_race = self.ax_race.bar(self.kb.classes, [0]*4, color=['#333333']*4)
        self.ax_race.set_ylim(0, 1.1)
        self.ax_race.set_title(f"Confidence (Threshold: {THRESHOLD})", fontsize=12)
        self.ax_race.axhline(y=THRESHOLD, color='red', linestyle='--', alpha=0.5)

        # AXIS 3: Clue List (Right - Dedicated Text Area)
        self.ax_text = self.fig.add_subplot(gs[2])
        self.ax_text.axis('off')
        self.ax_text.set_title("Clue History", fontsize=12, fontweight='bold')

        # Status Footer
        self.txt_status = self.fig.text(0.5, 0.02, "Initializing...", ha='center', fontsize=14, weight='bold')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

    def _on_key(self, event):
        if event.key in ['1', '2', '3', '4']:
            self.human_guess = int(event.key) - 1

    def update(self, state_vector, probs, status_msg, clue_list):
        # 1. Update Heatmap
        self.img_dream.set_data(state_vector.detach().numpy())
        
        # 2. Update Race Bars
        for i, bar in enumerate(self.bars_race):
            bar.set_height(probs[i])
            if probs[i] > THRESHOLD:
                bar.set_color('red') 
            else:
                bar.set_color('teal')

        # 3. Update Clue List (Render text lines)
        self.ax_text.clear()
        self.ax_text.axis('off')
        self.ax_text.set_title("Clue History", fontsize=12, fontweight='bold')
        
        y_pos = 0.90
        for i, clue in enumerate(clue_list):
            # Highlight the most recent clue
            weight = 'bold' if i == len(clue_list) - 1 else 'normal'
            color = 'blue' if i == len(clue_list) - 1 else 'black'
            
            self.ax_text.text(0.05, y_pos, f"{i+1}. {clue}", 
                              transform=self.ax_text.transAxes, 
                              fontsize=11, fontweight=weight, color=color)
            y_pos -= 0.08 # Spacing

        # 4. Status
        self.txt_status.set_text(status_msg)
        
        plt.pause(0.05) 

    def close(self):
        plt.close(self.fig)

    def show_result(self, winner_type, target_name):
        """
        Displays a dramatic visual overlay for the game result.
        winner_type: 'HUMAN', 'MACHINE', 'STALEMATE'
        """
        # 1. Determine Color Scheme
        if winner_type == 'HUMAN':
            bg_color = '#d4ffd4' # Light Green
            text_color = '#006400' # Dark Green
            main_text = "VICTORY!"
        elif winner_type == 'MACHINE':
            bg_color = '#ffd4d4' # Light Red
            text_color = '#8b0000' # Dark Red
            main_text = "DEFEAT"
        else:
            bg_color = '#eeeeee' # Grey
            text_color = 'black'
            main_text = "GAME OVER"

        # 2. Change Backgrounds
        self.fig.patch.set_facecolor(bg_color)
        self.ax_race.set_facecolor(bg_color)
        self.ax_text.set_facecolor(bg_color)
        
        # 3. Big Overlay Text
        self.fig.text(0.5, 0.5, main_text, 
                      ha='center', va='center', 
                      fontsize=50, fontweight='bold', color=text_color,
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=10))
        
        self.fig.text(0.5, 0.35, f"Identity: {target_name}", 
                      ha='center', va='center', 
                      fontsize=20, color='black')

        plt.draw()
        plt.pause(0.1)

# ==============================================================================
# MODULE 4: GAME ENGINE (Fixed Timing)
# ==============================================================================
class GameEngine:
    def __init__(self):
        self.kb = KnowledgeBase()
        self.physics = NeuralPhysics(self.kb.get_memory_matrix())
        self.ui = Visualizer(self.kb)
        
    def play_round(self):
        self.ui.setup()
        
        target_idx = random.randint(0, 3)
        target_name = self.kb.classes[target_idx]
        
        target_vec = self.kb.definitions[target_name]
        valid_clues = [k for k, v in target_vec.items() if v > 0.0]
        random.shuffle(valid_clues)
        
        print(f"\n>>> NEW GAME. Target Hidden. Press 1-4 to guess.")
        
        current_state = torch.zeros((1, DIM_SIZE))
        revealed_list = []
        
        game_over = False
        winner_type = None # 'HUMAN' or 'MACHINE'
        
        # --- MAIN LOOP ---
        for clue in valid_clues:
            if game_over: break
            
            # A. Reveal Clue (TEXT ONLY)
            revealed_list.append(clue)
            clues_str = "Clues: " + " | ".join(revealed_list[-5:]) 
            print(f"Clue Dropped: {clue}")
            
            # --- PHASE 1: READING DELAY (2 Seconds) ---
            # The clue is visible, but the physics haven't updated yet.
            # You can steal a win here if you are fast.
            start_read = time.time()
            read_duration = 2.0
            
            # Get current probability for UI continuity
            _, attn = self.physics.relax(current_state, step_size=0.0) 
            probs = attn.squeeze().tolist()

            while (time.time() - start_read) < read_duration:
                remaining = read_duration - (time.time() - start_read)
                status_msg = f"Clue Dropped... Processing in {remaining:.1f}s"
                
                # Update UI (Show clue, but keep old state)
                self.ui.update(current_state, probs, status_msg, revealed_list)
                
                # Check Input
                if self.ui.human_guess is not None:
                    if self.ui.human_guess == target_idx:
                        winner_type = 'HUMAN'
                    else:
                        winner_type = 'MACHINE' # Wrong guess = loss
                    game_over = True
                    break
            
            if game_over: break

            # --- PHASE 2: INJECTION ---
            # Now we actually add the vector to the state
            clue_vec = self.kb.get_clue_vector(clue)
            current_state = current_state + clue_vec
            
            # --- PHASE 3: DRIFTING (Physics Calculation) ---
            start_time = time.time()
            turn_duration = 5.0 
            
            while (time.time() - start_time) < turn_duration:
                
                # Physics: Relax
                new_state, attn = self.physics.relax(current_state, step_size=0.005)
                probs = attn.squeeze().tolist()
                
                current_state = new_state 
                
                elapsed = time.time() - start_time
                status_msg = f"Analyzing... ({elapsed:.1f}s / {turn_duration:.1f}s)"
                
                self.ui.update(current_state, probs, status_msg, revealed_list)
                
                # Check Input
                if self.ui.human_guess is not None:
                    print(f">>> REGISTERED GUESS: {self.kb.classes[self.ui.human_guess]}")
                    if self.ui.human_guess == target_idx:
                        winner_type = 'HUMAN'
                    else:
                        winner_type = 'MACHINE'
                    game_over = True
                    break
                
                # Check AI Confidence
                max_conf = max(probs)
                if max_conf > THRESHOLD:
                    ai_idx = probs.index(max_conf)
                    print(f">>> AI BUZZED IN: {self.kb.classes[ai_idx]}")
                    if ai_idx == target_idx:
                        winner_type = 'MACHINE'
                    else:
                        # If AI hallucinates, Human wins
                        winner_type = 'HUMAN' 
                    game_over = True
                    break
            
            if game_over: break

        # END OF ROUND
        if not winner_type: winner_type = 'MACHINE' # Ran out of clues
        
        print(f"\n>>> GAME OVER. Result: {winner_type}")
        print(f">>> Target was: {target_name}")
        
        # Trigger the Victory/Defeat Screen
        self.ui.show_result(winner_type, target_name)
        
        # Pause to let it soak in
        plt.pause(3.0) 
        self.ui.close()

if __name__ == "__main__":
    game = GameEngine()
    while True:
        game.play_round()
        if input("Play Again? (y/n): ").lower() != 'y': break