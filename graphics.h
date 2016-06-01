// ############################################
//       UVSQ -- Licence UFR des Sciences
//
// L1S1 IN100 -- Fondements de l'Informatique 1
// L1S2 IN210 -- Fondements de l'Informatique 2
//
// Franck QUESSETTE -- Franck.Quessette@uvsq.fr
// Version de novembre 2009
//
// Modifications E. Villani (2012) et B. Charbord (2013)
//
// SOMMAIRE
//
// 1. TYPES, VARIABLES, CONSTANTES ... ligne  39
// 2. AFFICHAGE ...................... ligne  76
// 3. GESTION D'�V�NEMENTS ........... ligne  99
// 4. DESSIN D'OBJETS ................ ligne 135
// 5. �CRITURE DE TEXTE .............. ligne 180
// 6. LECTURE D'ENTIER ............... ligne 210
// 7. GESTION DU TEMPS ............... ligne 219
// 8. VALEUR AL�ATOIRES .............. ligne 241
// 9. RESEAUX ........................ ligne 256
// 10.DIVERS ......................... ligne 303
// #############################################


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <SDL/SDL.h>

// A decommenter si on fonctionne en local, sinon sur un terminal
// on a des lenteurs
// #define EN_LOCAL

// ###############################
// 1. TYPES, VARIABLES, CONSTANTES
// ###############################

// 1.1 TYPES

	// D�finition du type POINT
	typedef struct point {int x,y;} POINT;

	// D�finition du type COULEUR
	typedef Uint32 COULEUR;

	// Les bool�ens
	typedef int BOOL;
	
// 1.2 VARIABLES
	// Taille de l'affichage
	int WIDTH;
	int HEIGHT;

// 1.3 CONSTANTES

	// D�placement minimal lorsque l'on utilise les fl�ches.
	#define MINDEP 1

	// Constantes de couleur
	// 16 couleurs de base en francais

	#define argent     0xC0C0C0
	#define blanc      0xFFFFFF
	#define bleu       0x0000FF
	#define bleumarine 0x000080
	#define citronvert 0x00FF00
	#define cyan       0x00FFFF
	#define magenta    0xFF00FF
	#define gris       0x808080
	#define jaune      0xFFFF00
	#define marron     0x800000
	#define noir       0x000000
	#define rouge      0xFF0000
	#define sarcelle   0x008080
	#define vert       0x00FF00
	#define vertclair  0x008000
	#define vertolive  0x808000
	#define violet     0x800080

	// 140 couleurs en anglais
	#define aliceblue            0xF0F8FF
	#define antiquewhite         0xFAEBD7
	#define aqua                 0x00FFFF
	#define aquamarine           0x7FFFD4
	#define azure                0xF0FFFF
	#define beige                0xF5F5DC
	#define bisque               0xFFE4C4
	#define black                0x000000
	#define blanchedalmond       0xFFEBCD
	#define blue                 0x0000FF
	#define blueviolet           0x8A2BE2
	#define brown                0xA52A2A
	#define burlywood            0xDEB887
	#define cadetblue            0x5F9EA0
	#define chartreuse           0x7FFF00
	#define chocolate            0xD2691E
	#define coral                0xFF7F50
	#define cornflowerblue       0x6495ED
	#define cornsilk             0xFFF8DC
	#define crimson              0xDC143C
	#define cyan                 0x00FFFF
	#define darkblue             0x00008B
	#define darkcyan             0x008B8B
	#define darkgoldenrod        0xB8860B
	#define darkgray             0xA9A9A9
	#define darkgreen            0x006400
	#define darkkhaki            0xBDB76B
	#define darkmagenta          0x8B008B
	#define darkolivegreen       0x556B2F
	#define darkorange           0xFF8C00
	#define darkorchid           0x9932CC
	#define darkred              0x8B0000
	#define darksalmon           0xE9967A
	#define darkseagreen         0x8FBC8F
	#define darkslateblue        0x483D8B
	#define darkslategray        0x2F4F4F
	#define darkturquoise        0x00CED1
	#define darkviolet           0x9400D3
	#define deeppink             0xFF1493
	#define deepskyblue          0x00BFFF
	#define dimgray              0x696969
	#define dodgerblue           0x1E90FF
	#define firebrick            0xB22222
	#define floralwhite          0xFFFAF0
	#define forestgreen          0x228B22
	#define fuchsia              0xFF00FF
	#define gainsboro            0xDCDCDC
	#define ghostwhite           0xF8F8FF
	#define gold                 0xFFD700
	#define goldenrod            0xDAA520
	#define gray                 0x808080
	#define green                0x008000
	#define greenyellow          0xADFF2F
	#define honeydew             0xF0FFF0
	#define hotpink              0xFF69B4
	#define indianred            0xCD5C5C
	#define indigo               0x4B0082
	#define ivory                0xFFFFF0
	#define khaki                0xF0E68C
	#define lavender             0xE6E6FA
	#define lavenderblush        0xFFF0F5
	#define lawngreen            0x7CFC00
	#define lemonchiffon         0xFFFACD
	#define lightblue            0xADD8E6
	#define lightcoral           0xF08080
	#define lightcyan            0xE0FFFF
	#define lightgoldenrodyellow 0xFAFAD2
	#define lightgreen           0x90EE90
	#define lightgrey            0xD3D3D3
	#define lightpink            0xFFB6C1
	#define lightsalmon          0xFFA07A
	#define lightseagreen        0x20B2AA
	#define lightskyblue         0x87CEFA
	#define lightslategray       0x778899
	#define lightsteelblue       0xB0C4DE
	#define lightyellow          0xFFFFE0
	#define lime                 0x00FF00
	#define limegreen            0x32CD32
	#define linen                0xFAF0E6
	#define magenta              0xFF00FF
	#define maroon               0x800000
	#define mediumaquamarine     0x66CDAA
	#define mediumblue           0x0000CD
	#define mediumorchid         0xBA55D3
	#define mediumpurple         0x9370DB
	#define mediumseagreen       0x3CB371
	#define mediumslateblue      0x7B68EE
	#define mediumspringgreen    0x00FA9A
	#define mediumturquoise      0x48D1CC
	#define mediumvioletred      0xC71585
	#define midnightblue         0x191970
	#define mintcream            0xF5FFFA
	#define mistyrose            0xFFE4E1
	#define moccasin             0xFFE4B5
	#define navajowhite          0xFFDEAD
	#define navy                 0x000080
	#define oldlace              0xFDF5E6
	#define olive                0x808000
	#define olivedrab            0x6B8E23
	#define orange               0xFFA500
	#define orangered            0xFF4500
	#define orchid               0xDA70D6
	#define palegoldenrod        0xEEE8AA
	#define palegreen            0x98FB98
	#define paleturquoise        0xAFEEEE
	#define palevioletred        0xDB7093
	#define papayawhip           0xFFEFD5
	#define peachpuff            0xFFDAB9
	#define peru                 0xCD853F
	#define pink                 0xFFC0CB
	#define plum                 0xDDA0DD
	#define powderblue           0xB0E0E6
	#define purple               0x800080
	#define red                  0xFF0000
	#define rosybrown            0xBC8F8F
	#define royalblue            0x4169E1
	#define saddlebrown          0x8B4513
	#define salmon               0xFA8072
	#define sandybrown           0xF4A460
	#define seagreen             0x2E8B57
	#define seashell             0xFFF5EE
	#define sienna               0xA0522D
	#define silver               0xC0C0C0
	#define skyblue              0x87CEEB
	#define slateblue            0x6A5ACD
	#define slategray            0x708090
	#define snow                 0xFFFAFA
	#define springgreen          0x00FF7F
	#define steelblue            0x4682B4
	#define tan                  0xD2B48C
	#define teal                 0x008080
	#define thistle              0xD8BFD8
	#define tomato               0xFF6347
	#define turquoise            0x40E0D0
	#define violetlight          0xEE82EE
	#define wheat                0xF5DEB3
	#define white                0xFFFFFF
	#define whitesmoke           0xF5F5F5
	#define yellow               0xFFFF00
	#define yellowgreen          0x9ACD32


	// Constantes bool�ennes
	#define TRUE 1
	#define True 1
	#define true 1
	#define FALSE 0
	#define False 0
	#define false 0
    
    //Constante de reseau
    #define _TAILLE_RESEAU_TRANSFERT_ 1000

// ############
// 2. AFFICHAGE
// ############

	// 2.1 Initialisation de la fen�tre sur laquelle on dessine
	void init_graphics(int W, int H, char *titre);

	// 2.2 Affichage automatique ou manuel
	// Si l'affichage est automatique, chaque objet dessin�
	// est automatiquement affich�.
	// Sinon, il faut explicitement appeler la fonction
	// affiche_all() pour afficher les objets
	// synchro() est identique � affiche_all()
	void affiche_auto_on();
	void affiche_auto_off();
	void affiche_all();
	void synchro();

	// 2.3 Cr�ation de couleur
	// r g et b dans l'intervalle 0 .. 255
	COULEUR couleur_RGB(int r, int g, int b);


// #######################
// 3. GESTION D'�V�NEMENTS
// #######################

	// 3.1 Renvoie le ou les fl�ches appuy�es
	// sous forme d'un d�placement en 
	// x n�gatif = nombre d'appuis sur la fl�che gauche
	// x positif = nombre d'appuis sur la fl�che droite
	// y n�gatif = nombre d'appuis sur la fl�che bas
	// y positif = nombre d'appuis sur la fl�che haut
	// Instruction non bloquante, si aucune fl�che n'a �t�
	// appuy�e les champs x et y vaudront 0.
	POINT get_arrow();

	// 3.2 Renvoie d�placement de souris
	// Meme s�mantique que get_arrow()
	// Instruction non bloquante, si la souris n'a pas boug�,
	// les champs x et y vaudront 0.
	POINT get_mouse();

	// 3.3 Attend que l'on tape Echap et quitte
	// Instruction bloquante
	void wait_escape();

	// 3.4 Attend que l'utilisateur clique avec le bouton gauche
	// Renvoie les coordonn�es du point cliqu�
	// Instruction bloquante
	POINT wait_clic();

	// 3.5 Attend que l'on clique et renvoie dans button le bouton cliqu� :
	// *button vaut soit 'G' (pour Gauche), soit 'M' (pour milieu),
	// soit 'D' (pour Droit) en fonction du bouton cliqu�
	// Instruction bloquante
	POINT wait_clic_GMD(char *button);


// ##################
// 4. DESSIN D'OBJETS
// ##################

	// 4.1 Remplissage de tout l'�cran
	void fill_screen(COULEUR color);

	// 4.2 Dessine un pixel
	void draw_pixel(POINT p, COULEUR color);

	// 4.3 Dessine un segment
	void draw_line(POINT p1, POINT p2, COULEUR color);

	// 4.4 Dessine un rectangle non rempli
	// Les deux points sont deux points quelconques 
	// non adjacents du rectangle
	void draw_rectangle(POINT p1, POINT p2, COULEUR color);

	// 4.5 Dessine un rectangle rempli
	// Les deux point sont deux points quelconques 
	// non adjacents du rectangle
	void draw_fill_rectangle(POINT p1, POINT p2, COULEUR color);

	// 4.6 Dessine un cercle non rempli
	void draw_circle(POINT centre, int rayon, COULEUR color);

	// 4.7 Dessine un cercle rempli
	void draw_fill_circle(POINT centre, int rayon, COULEUR color);

	// 4.8 Dessine des quarts de cercle
	void draw_circle_HD(POINT centre, int rayon, COULEUR color);
	void draw_circle_BD(POINT centre, int rayon, COULEUR color);
	void draw_circle_HG(POINT centre, int rayon, COULEUR color);
	void draw_circle_BG(POINT centre, int rayon, COULEUR color);

	// 4.9 Dessine une ellipse remplie
	void draw_fill_ellipse(POINT F1, POINT F2, int r, COULEUR color);

	// 4.10 Dessine un triangle
	void draw_triangle(POINT p1, POINT p2, POINT p3, COULEUR color);

	// 4.11 Dessine un triangle rempli
	void draw_fill_triangle(POINT p1, POINT p2, POINT p3, COULEUR color);


// ####################
// 5. �CRITURE DE TEXTE
// ####################

// L'affichage se fait dans la fen�tre graphique si SDL_ttf est install�.
// Sinon il se fait dans dans la fen�tre shell

	// 5.1 Affiche du texte avec
	// Le texte est pass� dans l'argument "a_ecrire" 
	// la police est celle d�finie par la constante POLICE_NAME 
	//           dans graphics.c
	// la taille est pass�e en argument
	// l'argument p de type POINT est le point en haut � gauche
	// � partir duquel le texte s'affiche
	// la COULEUR C pass�e en argument est la couleur d'affichage
	void aff_pol(char *a_ecrire, int taille, POINT p, COULEUR C);

	// 5.2 Affiche un entier
	// Meme s�mantique que aff_pol()
	void aff_int(int n, int taille, POINT p, COULEUR C);

	// 5.3 Affiche dans la fen�tre graphique comme dans une fen�tre
	// shell mais sans d�roulement. Commence en haut et
	// se termine en bas.
	void write_text(char *a_ecrire);
	void write_int(int n);
	void write_bool(BOOL b);
	void writeln() ;


// ###################
// 6. LECTURE D'ENTIER
// ###################

	// 6.1 Renvoie l'entier tap� au clavier.
	// Cette fonction est bloquante
	int lire_entier_clavier();


// ###################
// 7. GESTION DU TEMPS
// ###################

	// 7.1 Chronom�tre �l�mentaire
	// D�clenchement du chrono
	// Le remet � z�ro s'il �tait d�j� lanc�
	void chrono_start();
	// Renvoie la valeur du chrono et ne l'arr�te pas
	float chrono_val();
	
	// 7.2 Attend le nombre de millisecondes pass� en argument
	void attendre(int millisecondes);

	// 7.3.1 Renvoie l'heure de l'heure courante
	int heure();
	// 7.3.2 Renvoie le nombre de minutes de l'heure courante
	int minute();
	// 7.3.3 Renvoie le nombre de secondes de l'heure courante
	int seconde();


// ####################
// 8. VALEUR AL�ATOIRES
// ####################

	// 8.1 Renvoie un float dans l'intervalle [0;1[
	float alea_float();
	// 8.2 Renvoie un int dans l'intervalle [0..N[
	// soit N valeurs diff�rentes de 0 � N-1
	int alea_int(int N);


// ##########
// 9. RESEAUX
// ##########
        
    // 9.1 Initialisation/Fermeture des fonctions reseau. A lancer avant toute fonction de reseau.
    void init_reseau();
    void fin_reseau();

    // 9.2 Ouverture d'un serveur au port indique
    void ouvrir_serveur( unsigned int port);

    
    // 9.3 Connexion (ou tentative) d'un client a l'adresse ip et au port indique
    // renvoie 1 en cas de connexion reussi, 0 sinon
    int connexion_client_serveur( char * adresse_ip, unsigned int port);
    
    // 9.4 Recherche et acceptation d'un client cherchant a se connecter, cote serveur.
    // Si aucune client n'est connecte, renvoie 0. 
    // Si un client est connecte, renvoie 1.
    int serveur_accepte_client();
    void serveur_ferme_client();
    
    // 9.5 Lecture d'un message. Renvoie le text envoy�.
    // Exemple de d�claration d'une variable qui contiendra la chaine de caract�re recue:
    // char* message_recu;
    char* lire_message();
    
    // 9.6 Ecriture d'un message. 
    // Le message a envoyer doit etre:
    //  - soit sous format d'une chaine de caract�re (exemple: "Coucou")
    //  - soit sous forme d'un tableau de caract�res comportant_TAILLE_RESEAU_TRASNFERT_ caracteres.
    // Exemple de d�claration d'une variable qui contiendra la chaine de caract�re � envoyer:
    // char message_a_envoyer[_TAILLE_RESEAU_TRASNFERT_]
    void envoyer_message( char* message);
    
    
    //9.7 Lecture/Ecriture d'un nombre.
    int lire_nombre();
    void envoyer_nombre (int nombre);
    
    //9.8 Recuperation de l'ip et du port du serveur ou du client
    const char*   ip_du_serveur(); 
    unsigned int port_du_serveur();
    const char*  ip_du_client(); 
    unsigned int port_du_client();
    
    
// ##########
// 10. DIVERS
// ##########

	// 9.1 Renvoie la distance entre deux points
	int distance(POINT P1, POINT P2);
