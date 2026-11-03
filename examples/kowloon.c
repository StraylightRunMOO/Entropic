#define WFC_IMPLEMENTATION
#define WFC_ENABLE_COPULA 1
#define WFC_ENABLE_BACKTRACK 1

#include "entropic/wfc.h"
#include "entropic/sobol.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ==========================================================================
 * Configuration Parameters
 * ========================================================================== */
typedef struct {
    uint32_t width;
    uint32_t depth;
    uint32_t floors;
    uint32_t courtyard_size;
    uint32_t seed;
} kwc_config_t;

kwc_config_t G_CFG = {
    .width = 32,
    .depth = 16,
    .floors = 1,
    .courtyard_size = 5,
    .seed = 1337
};

/* ==========================================================================
 * Tile Definitions (8 Room Classes)
 * ========================================================================== */
enum TileClasses {
    C_ALLEY    = 0,
    C_COURT    = 1,
    C_CLINIC   = 2,
    C_HOVEL    = 3,
    C_SHOP     = 4,
    C_MAHJONG  = 5,
    C_SHRINE   = 6,
    C_STAIRS   = 7
};

enum TileIDs {
    T_ALLEY          = 0,
    T_COURT_AIR      = 1,

    T_CLINIC_WAIT    = 2,
    T_CLINIC_EXAM    = 3,
    T_CLINIC_OP      = 4,
    T_CLINIC_OFF     = 5,

    T_HOVEL_DOOR     = 6,
    T_HOVEL_ROOM     = 7,

    T_SHOP_FRONT     = 8,
    T_SHOP_BACK      = 9,

    T_MAHJONG_ROOM   = 10,
    T_SHRINE_ALTAR   = 11,

    T_STAIR_UP       = 12,
    T_STAIR_DOWN     = 13,
    T_STAIR_BOTH     = 14
};

/* Output Formatting */
const char* TILE_SYMBOLS[] = {
    ".", " ",                           /* Alley, Air */
    "+", "e", "o", "x",                 /* Clinic */
    "H", "h",                           /* Hovel */
    "$", "s",                           /* Shop */
    "M", "&",                           /* Mahjong, Shrine */
    "^", "v", "I"                       /* Stairs */
};

const char* TILE_COLORS[] = {
    "\033[90m", "\033[0m",              /* Gray (Alley), None (Air) */
    "\033[91m", "\033[31m", "\033[31m", "\033[31m", /* Red (Clinic) */
    "\033[93m", "\033[33m",             /* Yellow (Hovel) */
    "\033[92m", "\033[32m",             /* Green (Shop) */
    "\033[94m", "\033[97m",             /* Blue (Mahjong), White (Shrine) */
    "\033[95m", "\033[95m", "\033[95m"  /* Magenta (Stairs) */
};

/* ==========================================================================
 * Helper: Add WFC Rules
 * ========================================================================== */
void add_bidirectional_rule(wfc_world_t* w, uint16_t t1, uint16_t t2, wfc_dir_t d) {
    wfc_add_rule(w, t1, t2, d);
    
    /* Reverse direction logic */
    wfc_dir_t rev;
    if (d == WFC_DIR_N) rev = WFC_DIR_S;
    else if (d == WFC_DIR_S) rev = WFC_DIR_N;
    else if (d == WFC_DIR_E) rev = WFC_DIR_W;
    else if (d == WFC_DIR_W) rev = WFC_DIR_E;
    else if (d == WFC_DIR_U) rev = WFC_DIR_D;
    else rev = WFC_DIR_U; // WFC_DIR_D
    
    wfc_add_rule(w, t2, t1, rev);
}

/* ==========================================================================
 * Architectural Grammar Builder
 * ========================================================================== */
void build_grammar(wfc_world_t* w) {
    /* Define Patterns 
     * FIX 2: Set T_COURT_AIR weight to 0.0f so the generation engine never 
     * spontaneously spawns new air shafts that vertically slice into POIs! */
    wfc_pattern_def_t patterns[] = {
        {T_ALLEY,        C_ALLEY,   15.0f}, 
        {T_COURT_AIR,    C_COURT,    0.0f}, /* Strict constraint: pre-seeded only */
        {T_CLINIC_WAIT,  C_CLINIC,   1.0f},
        {T_CLINIC_EXAM,  C_CLINIC,   1.0f},
        {T_CLINIC_OP,    C_CLINIC,   0.5f},
        {T_CLINIC_OFF,   C_CLINIC,   0.8f},
        {T_HOVEL_DOOR,   C_HOVEL,    5.0f},
        {T_HOVEL_ROOM,   C_HOVEL,   10.0f},
        {T_SHOP_FRONT,   C_SHOP,     3.0f},
        {T_SHOP_BACK,    C_SHOP,     2.0f},
        {T_MAHJONG_ROOM, C_MAHJONG,  2.0f},
        {T_SHRINE_ALTAR, C_SHRINE,   0.5f},
        {T_STAIR_UP,     C_STAIRS,   1.5f},
        {T_STAIR_DOWN,   C_STAIRS,   1.5f},
        {T_STAIR_BOTH,   C_STAIRS,   1.5f}
    };
    for (int i = 0; i < 15; i++) wfc_add_pattern(w, &patterns[i]);

    /* ==========================================
     * HORIZONTAL RULES (N, S, E, W)
     * ========================================== */
    wfc_dir_t horiz[] = {WFC_DIR_N, WFC_DIR_S, WFC_DIR_E, WFC_DIR_W};
    for (int i = 0; i < 4; i++) {
        wfc_dir_t d = horiz[i];

        /* 1. Core Infrastructure Symmetries */
        add_bidirectional_rule(w, T_ALLEY, T_ALLEY, d);
        add_bidirectional_rule(w, T_COURT_AIR, T_COURT_AIR, d);
        add_bidirectional_rule(w, T_COURT_AIR, T_ALLEY, d); 

        /* 2. UNIVERSAL ALLEY ESCAPE: Alleys can border ANYTHING.
         * This gives the solver a valid state to fall back on anywhere on the map. */
        for (uint16_t t = T_ALLEY; t <= T_STAIR_BOTH; t++) {
            add_bidirectional_rule(w, T_ALLEY, t, d);
            add_bidirectional_rule(w, T_COURT_AIR, t, d); // Courtyard can also touch any room boundary
        }

        /* 3. Fully Permissive Room Clustering */
        for (uint16_t t1 = T_CLINIC_WAIT; t1 <= T_STAIR_BOTH; t1++) {
            for (uint16_t t2 = T_CLINIC_WAIT; t2 <= T_STAIR_BOTH; t2++) {
                add_bidirectional_rule(w, t1, t2, d);
            }
        }

        /* 4. Cohesive internal structures */
        add_bidirectional_rule(w, T_MAHJONG_ROOM, T_MAHJONG_ROOM, d);
        add_bidirectional_rule(w, T_SHRINE_ALTAR, T_SHRINE_ALTAR, d);
        add_bidirectional_rule(w, T_STAIR_UP,     T_STAIR_UP, d);
        add_bidirectional_rule(w, T_STAIR_DOWN,   T_STAIR_DOWN, d);
        add_bidirectional_rule(w, T_STAIR_BOTH,   T_STAIR_BOTH, d);
        
        add_bidirectional_rule(w, T_CLINIC_WAIT,  T_CLINIC_EXAM, d);
        add_bidirectional_rule(w, T_CLINIC_WAIT,  T_CLINIC_OFF, d);
        add_bidirectional_rule(w, T_CLINIC_EXAM,  T_CLINIC_OP, d);
        add_bidirectional_rule(w, T_CLINIC_EXAM,  T_CLINIC_OFF, d);
        add_bidirectional_rule(w, T_CLINIC_EXAM,  T_CLINIC_EXAM, d);
        add_bidirectional_rule(w, T_CLINIC_OP,    T_CLINIC_OFF, d);
        add_bidirectional_rule(w, T_HOVEL_DOOR,   T_HOVEL_ROOM, d);
        add_bidirectional_rule(w, T_HOVEL_ROOM,   T_HOVEL_ROOM, d);
        add_bidirectional_rule(w, T_SHOP_FRONT,   T_SHOP_BACK, d);
        add_bidirectional_rule(w, T_SHOP_BACK,    T_SHOP_BACK, d);
    }

    /* ONLY enforce strict vertical stacking if the map actually has multiple floors */
    if (w->nz > 1) {
        wfc_add_rule(w, T_COURT_AIR, T_COURT_AIR, WFC_DIR_U);
        wfc_add_rule(w, T_COURT_AIR, T_COURT_AIR, WFC_DIR_D);

        add_bidirectional_rule(w, T_STAIR_UP,   T_STAIR_DOWN, WFC_DIR_U);
        add_bidirectional_rule(w, T_STAIR_BOTH, T_STAIR_BOTH, WFC_DIR_U);
        add_bidirectional_rule(w, T_STAIR_UP,   T_STAIR_BOTH, WFC_DIR_U);
        add_bidirectional_rule(w, T_STAIR_BOTH, T_STAIR_DOWN, WFC_DIR_U);

        for (uint16_t t_bottom = T_ALLEY; t_bottom <= T_SHRINE_ALTAR; t_bottom++) {
            if (t_bottom == T_COURT_AIR) continue;
            for (uint16_t t_top = T_ALLEY; t_top <= T_SHRINE_ALTAR; t_top++) {
                if (t_top == T_COURT_AIR) continue;
                wfc_add_rule(w, t_bottom, t_top, WFC_DIR_U);
                wfc_add_rule(w, t_top, t_bottom, WFC_DIR_D);
            }
        }
    }
}

/* ==========================================================================
 * Pre-Seed Map using Sobol & Hardcoding
 * ========================================================================== */
void force_cell(wfc_world_t* w, uint32_t x, uint32_t y, uint32_t z, uint16_t tile) {
    uint32_t idx = (z * w->ny + y) * w->nx + x;
    wfc_cell_t* cell = &w->cells[idx];
    
    /* Zero out mask and only allow our chosen tile */
    memset(cell->mask, 0, sizeof(cell->mask));
    cell->mask[tile >> 6] |= (1ULL << (tile & 63));

    /* FIX 4: Inform the solver engine that this cell has changed constraints 
     * so it fires initial propagation before deciding the first wavefront collapse step! */
    cell->dirty = 1;
}

void pre_collapse_seeds(wfc_world_t* w) {
    uint32_t cx = G_CFG.width / 2;
    uint32_t cy = G_CFG.depth / 2;
    uint32_t half_c = G_CFG.courtyard_size / 2;

    /* 1. Clear out the center courtyard void across all floors */
    for (uint32_t z = 0; z < G_CFG.floors; z++) {
        for (uint32_t y = cy - half_c; y < cy + half_c; y++) {
            for (uint32_t x = cx - half_c; x < cx + half_c; x++) {
                force_cell(w, x, y, z, T_COURT_AIR);
            }
        }
    }

    /* 2. Seed POIs using the corrected Sobol Sequence execution */
    sob31_generator_t sob;
    if (sob31_init(&sob, 3)) { 
        sob31_next(&sob, 0); sob31_next(&sob, 1); sob31_next(&sob, 2);
        sob.count++; 

        int seeded_count = 0;
        int safety_timeout = 0;
        int target_pois = 8; 

        while (seeded_count < target_pois && safety_timeout < 100) {
            safety_timeout++;

            uint64_t raw_x = sob31_next(&sob, 0);
            uint64_t raw_y = sob31_next(&sob, 1);
            uint64_t raw_z = sob31_next(&sob, 2);
            sob.count++;

            uint32_t px = (uint32_t)(sob31_to_double(raw_x) * G_CFG.width);
            uint32_t py = (uint32_t)(sob31_to_double(raw_y) * G_CFG.depth);
            uint32_t pz = (uint32_t)(sob31_to_double(raw_z) * G_CFG.floors);

            /* Safety Checks to Prevent Contradictions: */
            if (px >= cx - half_c && px < cx + half_c && py >= cy - half_c && py < cy + half_c) {
                continue; 
            }
            if (px <= 1 || px >= G_CFG.width - 2 || py <= 1 || py >= G_CFG.depth - 2) {
                continue;
            }

            uint16_t target_tile = (seeded_count % 2 == 0) ? T_CLINIC_WAIT : T_SHRINE_ALTAR;
            force_cell(w, px, py, pz, target_tile);
            seeded_count++;
        }
        sob31_free(&sob);
    }
}

/* ==========================================================================
 * Output Generators
 * ========================================================================== */
void print_ascii_map(wfc_world_t* w) {
    printf("\n=== KOWLOON WALLED CITY GENERATOR ===\n");
    for (uint32_t z = 0; z < w->nz; z++) {
        printf("\n[ FLOOR %d ]\n", z + 1);
        for (uint32_t y = 0; y < w->ny; y++) {
            for (uint32_t x = 0; x < w->nx; x++) {
                uint16_t tile = wfc_get_cell_pattern(w, x, y, z);
                if (tile <= T_STAIR_BOTH) {
                    printf("%s%s\033[0m", TILE_COLORS[tile], TILE_SYMBOLS[tile]);
                } else {
                    printf("?");
                }
            }
            printf("\n");
        }
    }

    printf("\n--- LEGEND ---\n");
    printf("\033[90m.\033[0m Alley / Corridors     | \033[91m+ \033[31me o x\033[0m Clinic (Wait, Exam, Op, Office)\n");
    printf("\033[93mH \033[33mh\033[0m Hovel (Door, Room)  | \033[92m$ \033[32ms\033[0m Shop (Front, Back)\n");
    printf("\033[94mM\033[0m Mahjong Parlor          | \033[97m&\033[0m Shrine Altar\n");
    printf("\033[95m^ v I\033[0m Stairs (Up, Down, Both) | (Space) Central Courtyard / Air Shaft\n\n");
}

void export_to_json(wfc_world_t* w, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) return;

    fprintf(f, "{\n  \"width\": %d,\n  \"depth\": %d,\n  \"floors\": %d,\n  \"grid\": [\n", 
            w->nx, w->ny, w->nz);

    for (uint32_t z = 0; z < w->nz; z++) {
        fprintf(f, "    [\n");
        for (uint32_t y = 0; y < w->ny; y++) {
            fprintf(f, "      [");
            for (uint32_t x = 0; x < w->nx; x++) {
                fprintf(f, "%d%s", wfc_get_cell_pattern(w, x, y, z), (x == w->nx - 1) ? "" : ", ");
            }
            fprintf(f, "]%s\n", (y == w->ny - 1) ? "" : ",");
        }
        fprintf(f, "    ]%s\n", (z == w->nz - 1) ? "" : ",");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
    printf("Successfully wrote structural layout to %s\n", filename);
}

/* ==========================================================================
 * Main Routine
 * ========================================================================== */
int main(void) {
    wfc_config_t cfg = {
        .nx = G_CFG.width,
        .ny = G_CFG.depth,
        .nz = G_CFG.floors,
        .seed = G_CFG.seed,
        .seamless = 0
    };

    wfc_world_t* w = wfc_world_create(&cfg);
    if (!w) {
        printf("Fatal: Failed to allocate WFC world.\n");
        return 1;
    }

    printf("Building Architectural Grammar...\n");
    build_grammar(w);

    printf("Pre-seeding Sobol POIs and Courtyard Carve-out...\n");
    pre_collapse_seeds(w);

    printf("Collapsing WFC Wavefront...\n");
    wfc_err_t err = wfc_collapse(w);
    if (err != WFC_OK) {
        printf("Collapse Contradiction Failed (Error %d).\n", err);
        print_ascii_map(w);
    } else {
        print_ascii_map(w);
        export_to_json(w, "kowloon_layout.json");
    }

    wfc_world_destroy(w);
    return 0;
}

