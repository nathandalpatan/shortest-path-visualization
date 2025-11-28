import pygame
import heapq
from collections import deque
import math
import random

pygame.init()

# Constants
WIDTH, HEIGHT = 800, 900
ROWS = 40
COLS = 40
CELL_SIZE = WIDTH // COLS
INFO_HEIGHT = 100

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)
LIGHT_GREY = (200, 200, 200)

# Cell types
EMPTY = 0
WALL = 1
START = 2
END = 3
PATH = 4
VISITED = 5
WEIGHT = 6

class Cell:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.type = EMPTY
        self.weight = 1

    def get_color(self):
        if self.type == EMPTY:
            return WHITE
        elif self.type == WALL:
            return BLACK
        elif self.type == START:
            return GREEN
        elif self.type == END:
            return RED
        elif self.type == PATH:
            return PURPLE
        elif self.type == VISITED:
            return TURQUOISE
        elif self.type == WEIGHT:
            return ORANGE
        return WHITE

# ====================
# PATHFINDING ALGORITHMS
# ====================

class PathfindingAlgorithms:
    def __init__(self, grid, start, end, visualizer):
        self.grid = grid
        self.start = start
        self.end = end
        self.visualizer = visualizer
        self.rows = len(grid)
        self.cols = len(grid[0])

    def get_neighbors(self, row, col):
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                if self.grid[new_row][new_col].type != WALL:
                    neighbors.append((new_row, new_col))
        return neighbors

    def heuristic(self, row1, col1, row2, col2):
        return abs(row1 - row2) + abs(col1 - col2)

    def reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            current = came_from[current]
            if self.grid[current[0]][current[1]].type not in [START, END]:
                self.grid[current[0]][current[1]].type = PATH
                path.append(current)
        return path

    def check_cancel(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.visualizer.running = False
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

    def visualize_step(self):
        if self.check_cancel():
            return True
        self.visualizer.draw_grid()
        self.visualizer.draw_info()
        pygame.display.flip()
        self.visualizer.clock.tick(60)
        return False

    def a_star(self):
        if not self.start or not self.end:
            return False

        start_row, start_col = self.start
        end_row, end_col = self.end

        open_set = [(0, start_row, start_col)]
        came_from = {}
        g_score = {(start_row, start_col): 0}
        f_score = {(start_row, start_col): self.heuristic(start_row, start_col, end_row, end_col)}

        while open_set:
            if self.visualize_step():
                return False

            _, row, col = heapq.heappop(open_set)

            if (row, col) == (end_row, end_col):
                self.reconstruct_path(came_from, (row, col))
                return True

            for n_row, n_col in self.get_neighbors(row, col):
                cell = self.grid[n_row][n_col]
                weight = cell.weight if cell.type == WEIGHT else 1
                temp_g = g_score.get((row, col), float('inf')) + weight

                if temp_g < g_score.get((n_row, n_col), float('inf')):
                    came_from[(n_row, n_col)] = (row, col)
                    g_score[(n_row, n_col)] = temp_g
                    f_score[(n_row, n_col)] = temp_g + self.heuristic(n_row, n_col, end_row, end_col)
                    heapq.heappush(open_set, (f_score[(n_row, n_col)], n_row, n_col))

                    if self.grid[n_row][n_col].type == EMPTY:
                        self.grid[n_row][n_col].type = VISITED

        return False

    def dijkstra(self):
        if not self.start or not self.end:
            return False

        start_row, start_col = self.start
        end_row, end_col = self.end

        open_set = [(0, start_row, start_col)]
        came_from = {}
        distance = {(start_row, start_col): 0}

        while open_set:
            if self.visualize_step():
                return False

            dist, row, col = heapq.heappop(open_set)

            if (row, col) == (end_row, end_col):
                self.reconstruct_path(came_from, (row, col))
                return True

            for n_row, n_col in self.get_neighbors(row, col):
                cell = self.grid[n_row][n_col]
                weight = cell.weight if cell.type == WEIGHT else 1
                new_dist = distance[(row, col)] + weight

                if new_dist < distance.get((n_row, n_col), float('inf')):
                    distance[(n_row, n_col)] = new_dist
                    came_from[(n_row, n_col)] = (row, col)
                    heapq.heappush(open_set, (new_dist, n_row, n_col))

                    if self.grid[n_row][n_col].type == EMPTY:
                        self.grid[n_row][n_col].type = VISITED

        return False

    def bfs(self):
        if not self.start or not self.end:
            return False

        start_row, start_col = self.start
        end_row, end_col = self.end

        queue = deque([(start_row, start_col)])
        came_from = {}
        visited = {(start_row, start_col)}

        while queue:
            if self.visualize_step():
                return False

            row, col = queue.popleft()

            if (row, col) == (end_row, end_col):
                self.reconstruct_path(came_from, (row, col))
                return True

            for n_row, n_col in self.get_neighbors(row, col):
                if (n_row, n_col) not in visited:
                    visited.add((n_row, n_col))
                    came_from[(n_row, n_col)] = (row, col)
                    queue.append((n_row, n_col))

                    if self.grid[n_row][n_col].type == EMPTY:
                        self.grid[n_row][n_col].type = VISITED

        return False

    def dfs(self):
        if not self.start or not self.end:
            return False

        start_row, start_col = self.start
        end_row, end_col = self.end

        stack = [(start_row, start_col)]
        came_from = {}
        visited = {(start_row, start_col)}

        while stack:
            if self.visualize_step():
                return False

            row, col = stack.pop()

            if (row, col) == (end_row, end_col):
                self.reconstruct_path(came_from, (row, col))
                return True

            for n_row, n_col in self.get_neighbors(row, col):
                if (n_row, n_col) not in visited:
                    visited.add((n_row, n_col))
                    came_from[(n_row, n_col)] = (row, col)
                    stack.append((n_row, n_col))

                    if self.grid[n_row][n_col].type == EMPTY:
                        self.grid[n_row][n_col].type = VISITED

        return False

    def greedy_best_first(self):
        if not self.start or not self.end:
            return False

        start_row, start_col = self.start
        end_row, end_col = self.end

        open_set = [(self.heuristic(start_row, start_col, end_row, end_col), start_row, start_col)]
        came_from = {}
        visited = set()

        while open_set:
            if self.visualize_step():
                return False

            _, row, col = heapq.heappop(open_set)

            if (row, col) in visited:
                continue

            visited.add((row, col))

            if (row, col) == (end_row, end_col):
                self.reconstruct_path(came_from, (row, col))
                return True

            for n_row, n_col in self.get_neighbors(row, col):
                if (n_row, n_col) not in visited:
                    came_from[(n_row, n_col)] = (row, col)
                    h = self.heuristic(n_row, n_col, end_row, end_col)
                    heapq.heappush(open_set, (h, n_row, n_col))

                    if self.grid[n_row][n_col].type == EMPTY:
                        self.grid[n_row][n_col].type = VISITED

        return False

    def bidirectional_bfs(self):
        if not self.start or not self.end:
            return False

        start_row, start_col = self.start
        end_row, end_col = self.end

        queue_start = deque([(start_row, start_col)])
        queue_end = deque([(end_row, end_col)])

        came_from_start = {}
        came_from_end = {}

        visited_start = {(start_row, start_col)}
        visited_end = {(end_row, end_col)}

        def reconstruct_bidirectional_path(meeting_point):
            path = []
            current = meeting_point
            while current in came_from_start:
                current = came_from_start[current]
                if self.grid[current[0]][current[1]].type not in [START, END]:
                    self.grid[current[0]][current[1]].type = PATH
                    path.append(current)

            current = meeting_point
            while current in came_from_end:
                current = came_from_end[current]
                if self.grid[current[0]][current[1]].type not in [START, END]:
                    self.grid[current[0]][current[1]].type = PATH
                    path.append(current)
            return path

        while queue_start and queue_end:
            if self.visualize_step():
                return False

            # Expand from start
            if queue_start:
                row, col = queue_start.popleft()

                if (row, col) in visited_end:
                    reconstruct_bidirectional_path((row, col))
                    return True

                for n_row, n_col in self.get_neighbors(row, col):
                    if (n_row, n_col) not in visited_start:
                        visited_start.add((n_row, n_col))
                        came_from_start[(n_row, n_col)] = (row, col)
                        queue_start.append((n_row, n_col))

                        if self.grid[n_row][n_col].type == EMPTY:
                            self.grid[n_row][n_col].type = VISITED

            # Expand from end
            if queue_end:
                row, col = queue_end.popleft()

                if (row, col) in visited_start:
                    reconstruct_bidirectional_path((row, col))
                    return True

                for n_row, n_col in self.get_neighbors(row, col):
                    if (n_row, n_col) not in visited_end:
                        visited_end.add((n_row, n_col))
                        came_from_end[(n_row, n_col)] = (row, col)
                        queue_end.append((n_row, n_col))

                        if self.grid[n_row][n_col].type == EMPTY:
                            self.grid[n_row][n_col].type = VISITED

        return False

    def swarm_algorithm(self):
        if not self.start or not self.end:
            return False

        start_row, start_col = self.start
        end_row, end_col = self.end

        # Initialize particles
        num_particles = 20
        particles = []

        for _ in range(num_particles):
            particles.append({
                'pos': [start_row, start_col],
                'velocity': [random.uniform(-1, 1), random.uniform(-1, 1)],
                'best_pos': [start_row, start_col],
                'best_dist': float('inf')
            })

        global_best_pos = [start_row, start_col]
        global_best_dist = float('inf')

        max_iterations = 100
        came_from = {}

        for iteration in range(max_iterations):
            if self.visualize_step():
                return False

            for particle in particles:
                row, col = int(particle['pos'][0]), int(particle['pos'][1])

                if not (0 <= row < self.rows and 0 <= col < self.cols):
                    continue

                if self.grid[row][col].type == WALL:
                    continue

                dist = self.heuristic(row, col, end_row, end_col)

                if dist < particle['best_dist']:
                    particle['best_dist'] = dist
                    particle['best_pos'] = [row, col]

                if dist < global_best_dist:
                    global_best_dist = dist
                    global_best_pos = [row, col]

                if self.grid[row][col].type == EMPTY:
                    self.grid[row][col].type = VISITED

                if (row, col) == (end_row, end_col):
                    return True

                # Update velocity and position
                w = 0.5  # inertia
                c1 = 1.5  # cognitive
                c2 = 1.5  # social

                r1, r2 = random.random(), random.random()

                particle['velocity'][0] = (w * particle['velocity'][0] +
                                           c1 * r1 * (particle['best_pos'][0] - particle['pos'][0]) +
                                           c2 * r2 * (global_best_pos[0] - particle['pos'][0]))

                particle['velocity'][1] = (w * particle['velocity'][1] +
                                           c1 * r1 * (particle['best_pos'][1] - particle['pos'][1]) +
                                           c2 * r2 * (global_best_pos[1] - particle['pos'][1]))

                particle['pos'][0] += particle['velocity'][0]
                particle['pos'][1] += particle['velocity'][1]

        return False

    def convergent_swarm(self):
        # Similar to swarm but with convergence towards goal
        if not self.start or not self.end:
            return False

        start_row, start_col = self.start
        end_row, end_col = self.end

        open_set = [(self.heuristic(start_row, start_col, end_row, end_col), start_row, start_col)]
        came_from = {}
        visited = set()
        convergence_factor = 2.0

        while open_set:
            if self.visualize_step():
                return False

            _, row, col = heapq.heappop(open_set)

            if (row, col) in visited:
                continue

            visited.add((row, col))

            if (row, col) == (end_row, end_col):
                self.reconstruct_path(came_from, (row, col))
                return True

            neighbors = self.get_neighbors(row, col)

            # Sort neighbors by heuristic
            neighbors.sort(key=lambda n: self.heuristic(n[0], n[1], end_row, end_col))

            for n_row, n_col in neighbors:
                if (n_row, n_col) not in visited:
                    came_from[(n_row, n_col)] = (row, col)
                    h = self.heuristic(n_row, n_col, end_row, end_col) / convergence_factor
                    heapq.heappush(open_set, (h, n_row, n_col))

                    if self.grid[n_row][n_col].type == EMPTY:
                        self.grid[n_row][n_col].type = VISITED

        return False

# ====================
# MAIN VISUALIZER
# ====================

class PathfindingVisualizer:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Pathfinding Visualizer")
        self.clock = pygame.time.Clock()
        self.grid = [[Cell(r, c) for c in range(COLS)] for r in range(ROWS)]
        self.start = None
        self.end = None
        self.algorithms = [
            "A*", "Dijkstra", "BFS", "DFS", "Greedy",
            "Bidirectional", "Swarm", "Convergent Swarm"
        ]
        self.current_algo_index = 0
        self.algorithm = self.algorithms[self.current_algo_index]
        self.running = True
        self.searching = False
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

    def draw_grid(self):
        for row in self.grid:
            for cell in row:
                x = cell.col * CELL_SIZE
                y = cell.row * CELL_SIZE
                pygame.draw.rect(self.screen, cell.get_color(), (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(self.screen, LIGHT_GREY, (x, y, CELL_SIZE, CELL_SIZE), 1)

                if cell.type == WEIGHT:
                    weight_text = self.small_font.render(str(cell.weight), True, BLACK)
                    self.screen.blit(weight_text, (x + CELL_SIZE//3, y + CELL_SIZE//3))

    def draw_info(self):
        info_y = WIDTH
        pygame.draw.rect(self.screen, GREY, (0, info_y, WIDTH, INFO_HEIGHT))

        algo_text = self.font.render(f"Algorithm: {self.algorithm}", True, YELLOW)
        self.screen.blit(algo_text, (10, info_y + 5))

        instructions = [
            "L-Click: Wall | R-Click: Weight | S: Start | E: End | up arrow key/down arrow key: Weight +/-",
            "SPACE: Run | ESC: Cancel | C: Clear | R: Reset | TAB: Next Algo"
        ]

        for i, text in enumerate(instructions):
            inst_text = self.small_font.render(text, True, WHITE)
            self.screen.blit(inst_text, (10, info_y + 30 + i * 20))

        status = "Running... (Press ESC to cancel)" if self.searching else "Ready"
        status_color = RED if self.searching else GREEN
        status_text = self.small_font.render(status, True, status_color)
        self.screen.blit(status_text, (10, info_y + 75))

    def get_cell_from_pos(self, pos):
        x, y = pos
        if y >= WIDTH:
            return None, None
        row = y // CELL_SIZE
        col = x // CELL_SIZE
        if 0 <= row < ROWS and 0 <= col < COLS:
            return row, col
        return None, None

    def run_algorithm(self):
        self.searching = True
        algo = PathfindingAlgorithms(self.grid, self.start, self.end, self)

        result = False
        if self.algorithm == "A*":
            result = algo.a_star()
        elif self.algorithm == "Dijkstra":
            result = algo.dijkstra()
        elif self.algorithm == "BFS":
            result = algo.bfs()
        elif self.algorithm == "DFS":
            result = algo.dfs()
        elif self.algorithm == "Greedy":
            result = algo.greedy_best_first()
        elif self.algorithm == "Bidirectional":
            result = algo.bidirectional_bfs()
        elif self.algorithm == "Swarm":
            result = algo.swarm_algorithm()
        elif self.algorithm == "Convergent Swarm":
            result = algo.convergent_swarm()

        self.searching = False
        return result

    def clear_path(self):
        for row in self.grid:
            for cell in row:
                if cell.type in [VISITED, PATH]:
                    cell.type = EMPTY

    def reset(self):
        self.grid = [[Cell(r, c) for c in range(COLS)] for r in range(ROWS)]
        self.start = None
        self.end = None

    def next_algorithm(self):
        self.current_algo_index = (self.current_algo_index + 1) % len(self.algorithms)
        self.algorithm = self.algorithms[self.current_algo_index]

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if not self.searching:
                    if pygame.mouse.get_pressed()[0]:  # Left click
                        pos = pygame.mouse.get_pos()
                        row, col = self.get_cell_from_pos(pos)
                        if row is not None:
                            if self.grid[row][col].type not in [START, END]:
                                self.grid[row][col].type = WALL

                    if pygame.mouse.get_pressed()[2]:  # Right click
                        pos = pygame.mouse.get_pos()
                        row, col = self.get_cell_from_pos(pos)
                        if row is not None:
                            if self.grid[row][col].type not in [START, END]:
                                self.grid[row][col].type = WEIGHT
                                self.grid[row][col].weight = 5

                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                            pos = pygame.mouse.get_pos()
                            row, col = self.get_cell_from_pos(pos)
                            if row is not None:
                                if self.grid[row][col].type == WEIGHT:
                                    if event.key == pygame.K_UP:
                                        # Increment weight, max at 9
                                        self.grid[row][col].weight = min(self.grid[row][col].weight + 1, 9)
                                    elif event.key == pygame.K_DOWN:
                                        # Decrement weight, min at 1
                                        self.grid[row][col].weight = max(self.grid[row][col].weight - 1, 1)
                                        # If weight becomes 1, convert back to empty
                                        if self.grid[row][col].weight == 1:
                                            self.grid[row][col].type = EMPTY

                        elif event.key == pygame.K_s:
                            pos = pygame.mouse.get_pos()
                            row, col = self.get_cell_from_pos(pos)
                            if row is not None:
                                if self.start:
                                    self.grid[self.start[0]][self.start[1]].type = EMPTY
                                self.start = (row, col)
                                self.grid[row][col].type = START

                        elif event.key == pygame.K_e:
                            pos = pygame.mouse.get_pos()
                            row, col = self.get_cell_from_pos(pos)
                            if row is not None:
                                if self.end:
                                    self.grid[self.end[0]][self.end[1]].type = EMPTY
                                self.end = (row, col)
                                self.grid[row][col].type = END

                        elif event.key == pygame.K_SPACE:
                            self.clear_path()
                            self.run_algorithm()

                        elif event.key == pygame.K_c:
                            self.clear_path()

                        elif event.key == pygame.K_r:
                            self.reset()

                        elif event.key == pygame.K_TAB:
                            self.next_algorithm()

            self.screen.fill(WHITE)
            self.draw_grid()
            self.draw_info()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    visualizer = PathfindingVisualizer()
    visualizer.run()