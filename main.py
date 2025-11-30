import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import random
import time

# ==============================================================================
# MODULE 1: KNOWLEDGE BASE (Intensity Generator)
# ==============================================================================
class KnowledgeBase:
    def __init__(self):
        self.classes = ["Astronaut", "Diver", "Firefighter", "Hazmat"]
        
        # DEFINITIONS: The "Truth" of the universe.
        # These are the weights in the Hopfield Memory.
        self.weighted_definitions = {
            "Astronaut": {
                "Breathing": 0.5, "Radio": 0.9, "Dark": 0.3, "Cold": 0.5,
                "Slow": 1.0, "Drift": 1.0, "Helmet": 0.5, "Tether": 1.0,
                "Heat": 0.2, "Pressure": 0.0 # Vacuum is low pressure (handled via bipolar logic later if needed)
            },
            "Diver": {
                "Breathing": 0.9, "Radio": 0.2, "Dark": 0.9, "Cold": 0.8,
                "Slow": 0.7, "Drift": 0.8, "Helmet": 0.4, "Tether": 0.3,
                "Heat": 0.0, "Pressure": 1.0
            },
            "Firefighter": {
                "Breathing": 1.0, "Radio": 0.7, "Dark": 0.7, "Cold": -0.5,
                "Slow": -0.5, "Drift": -1.0, "Helmet": 0.7, "Tether": 0.0,
                "Heat": 1.0, "Pressure": 0.2
            },
            "Hazmat": {
                "Breathing": 0.6, "Radio": 0.4, "Dark": 0.1, "Cold": 0.2,
                "Slow": 0.2, "Drift": -1.0, "Helmet": 0.6, "Tether": 0.0,
                "Heat": 0.3, "Pressure": 0.0
            }
        }
        
        # TEXT GENERATOR: Maps (Feature, Intensity) -> String
        # Keys match the feature names above.
        # 0 = Low (0.1-0.4), 1 = Med (0.5-0.7), 2 = High (0.8-1.0)
        self.clue_descriptors = {
            "Breathing": ["Faint rhythm of breath", "Steady, mechanical breathing", "Loud, raspy gasping"],
            "Radio":     ["Occasional static burst", "Clear voice in ear", "Constant chatter/orders"],
            "Dark":      ["Shadows seem long here", "It is dim and murky", "Total, crushing blackness"],
            "Cold":      ["A bit chilly", "Numbing cold", "Bone-freezing absolute zero"],
            "Slow":      ["Movement is deliberate", "Resistance against limbs", "Dream-like slow motion"],
            "Drift":     ["Footing is unsure", "Floating slightly", "Complete suspension/Zero-G"],
            "Helmet":    ["Light headgear", "Sealed faceplate", "Heavy, reinforced helm"],
            "Tether":    ["Guide rope nearby", "Safety line attached", "Life depends on the line"],
            "Heat":      ["Temperature rising", "Sweltering humidity", "Searing, melting heat"],
            "Pressure":  ["Ears popping", "Heavy weight on chest", "Crushing depths"]
        }

        self.all_features = sorted(list(self.clue_descriptors.keys()))
        self.dim = len(self.all_features)

    def get_memory_matrix(self):
        """Creates the Bipolar Memory Matrix for the Brain"""
        matrix = []
        for name in self.classes:
            vec = torch.full((self.dim,), -0.1) # Baseline repulsion
            for f, w in self.weighted_definitions[name].items():
                if f in self.all_features:
                    vec[self.all_features.index(f)] = w
            matrix.append(vec)
        return torch.stack(matrix)

    def generate_clue_deck(self, target_name):
        """
        Generates clues based on the TARGET's reality.
        If Target=Firefighter, Heat=1.0 -> Generates High Intensity Heat clue.
        """
        deck = []
        target_traits = self.weighted_definitions[target_name]
        
        for feature, weight in target_traits.items():
            if weight <= 0.1: continue # Don't generate clues for non-existent things
            
            # Determine Intensity Bin
            if weight < 0.5: bin_idx = 0
            elif weight < 0.8: bin_idx = 1
            else: bin_idx = 2
            
            text = self.clue_descriptors[feature][bin_idx]
            
            # The Clue Object: (Feature Key, Vector Magnitude, Display Text)
            deck.append({
                "key": feature,
                "magnitude": weight, # The physics input matches the class reality
                "text": f"{text} (Intensity: {weight:.1f})" # Hint to user
            })
            
        random.shuffle(deck)
        return deck

    def get_projection_vector(self, clue_obj):
        """Creates the input vector for the physics engine"""
        vec = torch.zeros(self.dim)
        idx = self.all_features.index(clue_obj['key'])
        vec[idx] = clue_obj['magnitude'] # Input strength = Reality strength
        return vec

# ==============================================================================
# MODULE 2: NEURAL PHYSICS (Resonance & Noise Floor)
# ==============================================================================
class NeuralPhysics(torch.nn.Module):
    def __init__(self, memory_matrix, noise_floor=1.3, beta=3.0):
        super().__init__()
        self.memory = torch.nn.Parameter(memory_matrix, requires_grad=False)
        self.noise_floor = noise_floor # The Void Constant
        self.beta = beta

    def relax(self, current_state, step_size=0.01):
        # 1. Resonance (Dot Product)
        # We normalize Memory (The tuning fork) but NOT State (The sound volume)
        memory_norm = F.normalize(self.memory, dim=-1)
        resonance = torch.mm(current_state, memory_norm.T) 
        
        # 2. The Noise Floor (Fixed Void)
        # The Void has a constant energy. 
        # To win, a memory must resonate louder than the background noise.
        void_tensor = torch.tensor([[self.noise_floor]])
        logits = torch.cat([resonance, void_tensor], dim=1) 
        
        # 3. Attention
        attention = F.softmax(self.beta * logits, dim=-1)
        
        class_probs = attention[0, 0:4]
        unknown_prob = attention[0, 4].item()
        
        # 4. Reconstruction & Update
        target_state = torch.mm(class_probs.unsqueeze(0), self.memory)
        new_state = (1 - step_size) * current_state + (step_size * target_state)
        
        # 5. Signal-to-Noise Ratio (For Visuals)
        # How close is the leader to the noise floor?
        leader_score = torch.max(resonance).item()
        # Ratio: 0.0 = Far below floor. 1.0 = At floor. >1.0 = Breakout.
        # We invert it for the orb (1.0 = Big Orb/High Entropy)
        snr_ratio = max(0.0, 1.0 - (leader_score / self.noise_floor))
        
        return new_state, class_probs, unknown_prob, snr_ratio

# ==============================================================================
# MODULE 3: VISUALIZER (Orb tracks Signal-to-Noise)
# ==============================================================================
class Visualizer:
    def __init__(self, kb):
        self.kb = kb
        self.fig = None; self.ax_radar = None; self.orb = None; self.human_guess = None

    def setup(self):
        self.human_guess = None
        plt.close('all')
        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("The Semantic Race: Resonance")
        self.fig.patch.set_facecolor('white')
        
        self.ax_radar = self.fig.add_subplot(111, polar=True)
        
        num_vars = len(self.kb.classes)
        self.angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        self.angles += self.angles[:1]
        
        self.radar_poly, = self.ax_radar.plot([], [], color='teal', linewidth=2)
        self.radar_fill = self.ax_radar.fill([], [], color='teal', alpha=0.25)
        
        # Orb tracks the Gap between Signal and Noise Floor
        self.orb = self.ax_radar.scatter([0], [0], s=20000, color='gray', alpha=0.95, zorder=10)
        self.txt_orb = self.fig.text(0.5, 0.5, "NOISE", ha='center', va='center', color='white', weight='bold')
        
        labels = [f"{c}\n({i+1})" for i, c in enumerate(self.kb.classes)]
        self.ax_radar.set_xticks(self.angles[:-1])
        self.ax_radar.set_xticklabels(labels, size=11, weight='bold')
        self.ax_radar.set_yticks([])
        self.ax_radar.set_ylim(0, 1.0)
        
        self.txt_status = self.fig.text(0.5, 0.05, "Initializing...", ha='center', fontsize=14)
        self.txt_clues = self.fig.text(0.02, 0.98, "", va='top', fontsize=10, family='monospace', color='blue')
        
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

    def _on_key(self, event):
        if event.key in ['1', '2', '3', '4']:
            self.human_guess = int(event.key) - 1

    def update(self, probs, unknown_prob, snr_ratio, status_msg, clues_msg):
        # Radar
        disp_probs = probs / (np.sum(probs) + 1e-9)
        values = disp_probs.tolist() + disp_probs.tolist()[:1]
        self.radar_poly.set_data(self.angles, values)
        if len(self.ax_radar.collections) > 1: self.ax_radar.collections[-1].remove()
        self.ax_radar.fill(self.angles, values, color='teal', alpha=0.25)
        
        # Orb Size: Maps to SNR. 
        # If Signal > Noise Floor, Orb vanishes.
        orb_size = max(0, snr_ratio * 30000) 
        self.orb.set_sizes([orb_size])
        
        # Color: Grey -> Orange -> White
        if snr_ratio < 0.05: self.orb.set_color('white')
        elif snr_ratio < 0.3: self.orb.set_color('#ff9900') # Breakout imminent
        else: self.orb.set_color('gray')
        
        self.txt_orb.set_visible(snr_ratio > 0.1)
        self.txt_orb.set_text(f"{unknown_prob:.0%}\nUNK")

        self.txt_status.set_text(status_msg)
        self.txt_clues.set_text(clues_msg)
        plt.pause(0.02)

    def show_result(self, winner_type, target_name):
        color = '#d4ffd4' if winner_type == 'HUMAN' else '#ffd4d4'
        self.fig.patch.set_facecolor(color)
        self.ax_radar.set_facecolor(color)
        self.fig.text(0.5, 0.5, winner_type, ha='center', va='center', fontsize=40, weight='bold', color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        self.fig.text(0.5, 0.4, f"Identity: {target_name}", ha='center', fontsize=20)
        plt.draw()
        plt.pause(0.1)
    
    def close(self):
        plt.close(self.fig)

# ==============================================================================
# MODULE 4: GAME ENGINE (Resonance Loop)
# ==============================================================================
class GameEngine:
    def __init__(self):
        self.kb = KnowledgeBase()
        # Noise Floor = 1.2
        # A single 1.0 clue (High Intensity) isn't enough (1.0 < 1.2).
        # Two 0.7 clues (Medium) are enough (1.4 > 1.2).
        self.physics = NeuralPhysics(self.kb.get_memory_matrix(), noise_floor=1.2)
        self.ui = Visualizer(self.kb)
        
    def play_round(self):
        self.ui.setup()
        
        target_idx = random.randint(0, 3)
        target_name = self.kb.classes[target_idx]
        
        # Generate Deck based on Reality
        deck = self.kb.generate_clue_deck(target_name)
        
        print(f"\n>>> NEW GAME. Target Hidden. Press 1-4.")
        
        current_state = torch.zeros((1, self.kb.dim))
        revealed_list = []
        game_over = False
        winner = None
        
        for i, clue_obj in enumerate(deck):
            if game_over: break
            
            clue_text = clue_obj['text']
            revealed_list.append(clue_text)
            clues_str = "CLUES:\n" + "\n".join(revealed_list[-10:]) 
            print(f"Clue {i+1}: {clue_text}")
            
            # READ PHASE
            start_read = time.time()
            while (time.time() - start_read) < 1.5:
                _, attn, unk, snr = self.physics.relax(current_state, step_size=0.0)
                self.ui.update(attn.squeeze().numpy(), unk, snr, "Reading...", clues_str)
                if self.ui.human_guess is not None:
                    game_over = True; winner = self.check_human(target_idx); break
            
            if game_over: break

            # INJECT PHASE (Add Magnitude)
            current_state += self.kb.get_projection_vector(clue_obj)
            
            # DRIFT PHASE
            start_time = time.time()
            duration = 3.0
            
            while (time.time() - start_time) < duration:
                new_state, attn, unk_prob, snr = self.physics.relax(current_state, step_size=0.02)
                probs = attn.squeeze().tolist()
                current_state = new_state
                
                elapsed = time.time() - start_time
                self.ui.update(np.array(probs), unk_prob, snr, f"Resonating... {elapsed:.1f}s", clues_str)
                
                if self.ui.human_guess is not None:
                    game_over = True; winner = self.check_human(target_idx); break
                
                # --- RESONANCE CHECK ---
                # We trust the physics. If Unknown Prob drops below 20%, 
                # it means the signal has successfully pierced the Noise Floor.
                if unk_prob < 0.20:
                    ai_idx = probs.index(max(probs))
                    # Stricter margin check to prevent tie-breaking on noise
                    sorted_p = sorted(probs, reverse=True)
                    if (sorted_p[0] - sorted_p[1]) > 0.1:
                        print(f">>> AI BUZZ: {self.kb.classes[ai_idx]}")
                        if ai_idx == target_idx: winner = "MACHINE"
                        else: winner = "HUMAN"
                        game_over = True
                        break
            
            if game_over: break
            
        if not winner: winner = "MACHINE"
        self.ui.show_result(winner, target_name)
        plt.pause(3.0)
        self.ui.close()

    def check_human(self, target_idx):
        return "HUMAN" if self.ui.human_guess == target_idx else "MACHINE"

if __name__ == "__main__":
    game = GameEngine()
    while True:
        game.play_round()
        if input("Again? (y/n): ").lower() != 'y': break