/***************************************
Universidad Sim�n Bol�var
Departamento de Electr�nica y Circuitos
Algoritmos Gen�ticos ((EC - 5723))
Genetic Algorithm Filter Generator

C�sar Adolfo Pineda Carrero   - 15-11136
Andr�s Michele Scipione Damas - 16-11115
Oswaldo Monasterios Carrero   - 17-10392

Caracas 10/07/2023
***************************************/

/* Include of libraries for functions */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

/* Uncomment to enable DEBUG of functions */
//#define DEBUG_RANDOMIZE 
//#define DEBUG_GENE_SWAP
//#define DEBUG_MUTATION


/* Select Sorting Algorithm */
#define QUICKSORT
//#define BUBBLESORT

/* Mutation range */
#define MUTATION_PROB_R    10   //0.1% Prob
#define MUTATION_PROB_C    10   //0.1% Prob


/* Number of members of every gene */
#define NUMBER_OF_MEMEBERS          60
#define NUMBER_OF_BITS_PER_GENE     16
#define NUMBER_OF_DECADE_BITS       3
#define NUMBER_OF_VALUE_BITS        (NUMBER_OF_BITS_PER_GENE - NUMBER_OF_DECADE_BITS)

/* Top gene selection */
#define TOP_GENES          2
#define THRESHOLD_TO_RECOMBINE  30
#define NUMBER_OF_SONS_PER_TOP_GENE (NUMBER_OF_MEMEBERS/TOP_GENES)  //This value always must be an int

/* Mask used for Decade and Value */
#define DECADE_MASK  (0xFFFF >> NUMBER_OF_VALUE_BITS)    
#define VALUE_MASK   (~DECADE_MASK)


/* Frequency definitions */
#define MAX_FREQUENCY      100000  //100 KHz
#define DELTA_FREQUENCY    10      //10H z
#define NUMBER_STEPS       (MAX_FREQUENCY % DELTA_FREQUENCY)


/* Bitfield Masks for accessing */
#define BIT_0_MASK   0x0001
#define BIT_1_MASK   0x0002
#define BIT_2_MASK   0x0004
#define BIT_3_MASK   0x0008
#define BIT_4_MASK   0x0010
#define BIT_5_MASK   0x0020
#define BIT_6_MASK   0x0040
#define BIT_7_MASK   0x0080
#define BIT_8_MASK   0x0100
#define BIT_9_MASK   0x0200
#define BIT_10_MASK  0x0400
#define BIT_11_MASK  0x0800
#define BIT_12_MASK  0x1000
#define BIT_13_MASK  0x2000
#define BIT_14_MASK  0x4000
#define BIT_15_MASK  0x8000
#define TOTAL_16_BITS_MASK 0xFFFF


/* Macro Inline Function Definitions */
#define BIT_VALUE(value,mask)(value & mask ? 1:0)
#define VARIABLE_MASK(shift,LoR)(LoR == 0 ? TOTAL_16_BITS_MASK << shift : TOTAL_16_BITS_MASK >>shift)


/************ Structure definitions **********/

/* Union definition for genetic values */
typedef union{
    /* United value of expresion */
    uint16_t u16;

    /* Defined bit separated values for Genes*/
    struct{
        uint8_t decade      : 3;
        uint8_t value_8_LSB : 8;
        uint8_t value_5_MSB : 5;
    }gen_struct;

    /* Bitfield definition for mutations */
    struct {
        uint8_t bit_0  : 1;
        uint8_t bit_1  : 1;
        uint8_t bit_2  : 1;
        uint8_t bit_3  : 1;
        uint8_t bit_4  : 1;
        uint8_t bit_5  : 1;
        uint8_t bit_6  : 1;
        uint8_t bit_7  : 1;
        uint8_t bit_8  : 1;
        uint8_t bit_9  : 1;
        uint8_t bit_10 : 1;
        uint8_t bit_11 : 1;
        uint8_t bit_12 : 1;
        uint8_t bit_13 : 1;
        uint8_t bit_14 : 1;
        uint8_t bit_15 : 1;
    }u16_Bitfield;

}GENE_DEF;


/* Structure definition for Single Admitance populations and fitness */
typedef struct{
    GENE_DEF R_Gene;
    GENE_DEF C_Gene;
}FILTER_ADMITANCE_GENES;


/* Structure definition for filter generation */
typedef struct{
    float fitness;
    FILTER_ADMITANCE_GENES Admitance_1; //Admitance y1
    FILTER_ADMITANCE_GENES Admitance_2; //Admitance y2
    FILTER_ADMITANCE_GENES Admitance_3; //Admitance y3
    FILTER_ADMITANCE_GENES Admitance_4; //Admitance y4
    FILTER_ADMITANCE_GENES Admitance_5; //Admitance y5
    FILTER_ADMITANCE_GENES Admitance_6; //Admitance y6
} FILTER_CHROMOSOME;

/* typedef for filtering functions*/
typedef void* (*filter_Function)(int x);

/* Prototype Sorting functions definition */
void bubbleSort(FILTER_CHROMOSOME *population);
void quickSort(FILTER_CHROMOSOME *population, int left_limit, int right_limit);

/* Prototype Randomize functions definition*/
uint32_t randomize_32_t(void);
void population_Randomize(FILTER_CHROMOSOME *population);
void filter_Randomize(FILTER_CHROMOSOME *filter);

/* Prototype for transfer function evaluation */
//void TF_Evaluation(FILTER_POPULATION *population, filter_Function* func, bool inversor);

/* Genetic operation functions */
void gene_Swap(FILTER_CHROMOSOME *p_population, FILTER_CHROMOSOME *d_population);
void gene_Mutation(FILTER_CHROMOSOME *population);


/* Bitfield print function */
void print_Bitfield(int value);

/* Gene print function */
void print_Genes(FILTER_CHROMOSOME population, int number);
void print_Generation(FILTER_CHROMOSOME *population);

/* Filter Initialize function*/
void filter_Select(const char* filter_Name, int cutoff_1, int cutoff_2, int gain_1, int gain_2);

/* Filter Fitness calculate */
void fitness_Filter_Assign(FILTER_CHROMOSOME population);


/* Global variable definitions */
char input_Dummy[1];
int filter_Ideal_Values[NUMBER_STEPS];

int main(){
    /* Randomize of time variables */
    time_t t;
    srand((unsigned)time(&t));

    /* Genetic Filter Population */
    FILTER_CHROMOSOME Filter_Population_Parent [NUMBER_OF_MEMEBERS];
    FILTER_CHROMOSOME Filter_Population_Descendant[NUMBER_OF_MEMEBERS];

    /* Randomize initial Population */
    population_Randomize(Filter_Population_Parent);
    //filter_Randomize(&filter_1);

    printf("***********************\n");
    printf("Fitness Assigned\n\n");
    printf("Press Enter to continue...\n");
    scanf("%c",&input_Dummy);

    #ifdef QUICKSORT
        quickSort(Filter_Population_Parent, 0, NUMBER_OF_MEMEBERS - 1);
    #endif // QUICKSORT 

    #ifdef BUBBLESORT
        bubbleSort(Filter_Population_Parent);
    #endif // QUICKSORT 


    //print_Generation(Filter_Population_Parent);


    gene_Swap(Filter_Population_Parent, Filter_Population_Descendant);
    gene_Mutation(Filter_Population_Descendant);
    //printf("Descendant\n\n");
    //print_Genes(&descendant_Population);

    //gene_Mutation(&parent_Population);

    printf("Press Enter to Exit...\n");
    scanf("%c",&input_Dummy);

    return 0;
}


void bubbleSort(FILTER_CHROMOSOME *population){

    int i, j;
    bool swapped;
    float aux;
    FILTER_ADMITANCE_GENES aux1, aux2, aux3, aux4, aux5, aux6 ;

    for (i = 0; i < NUMBER_OF_MEMEBERS - 1; i++) {

        swapped = false;

        for (j = 0; j < NUMBER_OF_MEMEBERS - i - 1; j++) {
            if (population[j].fitness > population[j + 1].fitness) {

                aux = population[j].fitness;

                aux1.R_Gene.u16 = population[j].Admitance_1.R_Gene.u16;
                aux1.C_Gene.u16 = population[j].Admitance_1.C_Gene.u16;

                aux2.R_Gene.u16 = population[j].Admitance_2.R_Gene.u16;
                aux2.C_Gene.u16 = population[j].Admitance_2.C_Gene.u16;

                aux3.R_Gene.u16 = population[j].Admitance_3.R_Gene.u16;
                aux3.C_Gene.u16 = population[j].Admitance_3.C_Gene.u16;

                aux4.R_Gene.u16 = population[j].Admitance_4.R_Gene.u16;
                aux4.C_Gene.u16 = population[j].Admitance_4.C_Gene.u16;

                aux5.R_Gene.u16 = population[j].Admitance_5.R_Gene.u16;
                aux5.C_Gene.u16 = population[j].Admitance_5.C_Gene.u16;

                aux6.R_Gene.u16 = population[j].Admitance_6.R_Gene.u16;
                aux6.C_Gene.u16 = population[j].Admitance_6.C_Gene.u16;

                population[j].fitness = population[j + 1].fitness;

                population[j].Admitance_1.R_Gene.u16 = population[j+1].Admitance_1.R_Gene.u16;
                population[j].Admitance_1.C_Gene.u16 = population[j+1].Admitance_1.C_Gene.u16;
                                                                   
                population[j].Admitance_2.R_Gene.u16 = population[j+1].Admitance_2.R_Gene.u16;
                population[j].Admitance_2.C_Gene.u16 = population[j+1].Admitance_2.C_Gene.u16;
                                                                   
                population[j].Admitance_3.R_Gene.u16 = population[j+1].Admitance_3.R_Gene.u16;
                population[j].Admitance_3.C_Gene.u16 = population[j+1].Admitance_3.C_Gene.u16;
                                                                   
                population[j].Admitance_4.R_Gene.u16 = population[j+1].Admitance_4.R_Gene.u16;
                population[j].Admitance_4.C_Gene.u16 = population[j+1].Admitance_4.C_Gene.u16;
                                                                   
                population[j].Admitance_5.R_Gene.u16 = population[j+1].Admitance_5.R_Gene.u16;
                population[j].Admitance_5.C_Gene.u16 = population[j+1].Admitance_5.C_Gene.u16;
                                                                   
                population[j].Admitance_6.R_Gene.u16 = population[j+1].Admitance_6.R_Gene.u16;
                population[j].Admitance_6.C_Gene.u16 = population[j+1].Admitance_6.C_Gene.u16;


                population[j + 1].fitness = aux;

                population[j + 1].Admitance_1.R_Gene.u16 = aux1.R_Gene.u16;
                population[j + 1].Admitance_1.C_Gene.u16 = aux1.C_Gene.u16;
                                                         
                population[j + 1].Admitance_2.R_Gene.u16 = aux2.R_Gene.u16;
                population[j + 1].Admitance_2.C_Gene.u16 = aux2.C_Gene.u16;
                                                         
                population[j + 1].Admitance_3.R_Gene.u16 = aux3.R_Gene.u16;
                population[j + 1].Admitance_3.C_Gene.u16 = aux3.C_Gene.u16;
                                                         
                population[j + 1].Admitance_4.R_Gene.u16 = aux4.R_Gene.u16;
                population[j + 1].Admitance_4.C_Gene.u16 = aux4.C_Gene.u16;
                                                         
                population[j + 1].Admitance_5.R_Gene.u16 = aux5.R_Gene.u16;
                population[j + 1].Admitance_5.C_Gene.u16 = aux5.C_Gene.u16;
                                                         
                population[j + 1].Admitance_6.R_Gene.u16 = aux6.R_Gene.u16;
                population[j + 1].Admitance_6.C_Gene.u16 = aux6.C_Gene.u16;

                swapped = true;

            }
        }


        if (swapped == false) {
            break;
        }

    }

}



void quickSort(FILTER_CHROMOSOME *population, int left_limit, int right_limit){

    int left, right;
    float pivot, aux;
    FILTER_ADMITANCE_GENES aux1, aux2, aux3, aux4, aux5, aux6;

    left = left_limit;
    right = right_limit;

    pivot = population[(left + right) / 2].fitness;

    do{

        while((population[left].fitness < pivot) && (left < right_limit)) {
            left++;
        }

        while((pivot < population[right].fitness) && (right > left_limit)) {
            right--;
        }

        if(left <= right){

                aux = population[left].fitness;

                aux1.R_Gene.u16 = population[left].Admitance_1.R_Gene.u16;
                aux1.C_Gene.u16 = population[left].Admitance_1.C_Gene.u16;

                aux2.R_Gene.u16 = population[left].Admitance_2.R_Gene.u16;
                aux2.C_Gene.u16 = population[left].Admitance_2.C_Gene.u16;

                aux3.R_Gene.u16 = population[left].Admitance_3.R_Gene.u16;
                aux3.C_Gene.u16 = population[left].Admitance_3.C_Gene.u16;
                                            
                aux4.R_Gene.u16 = population[left].Admitance_4.R_Gene.u16;
                aux4.C_Gene.u16 = population[left].Admitance_4.C_Gene.u16;
                                          
                aux5.R_Gene.u16 = population[left].Admitance_5.R_Gene.u16;
                aux5.C_Gene.u16 = population[left].Admitance_5.C_Gene.u16;
                                          
                aux6.R_Gene.u16 = population[left].Admitance_6.R_Gene.u16;
                aux6.C_Gene.u16 = population[left].Admitance_6.C_Gene.u16;


                population[left].fitness = population[right].fitness;

                population[left].Admitance_1.R_Gene.u16 = population[right].Admitance_1.R_Gene.u16;
                population[left].Admitance_1.C_Gene.u16 = population[right].Admitance_1.C_Gene.u16;
                                                                   
                population[left].Admitance_2.R_Gene.u16 = population[right].Admitance_2.R_Gene.u16;
                population[left].Admitance_2.C_Gene.u16 = population[right].Admitance_2.C_Gene.u16;
                                                             
                population[left].Admitance_3.R_Gene.u16 = population[right].Admitance_3.R_Gene.u16;
                population[left].Admitance_3.C_Gene.u16 = population[right].Admitance_3.C_Gene.u16;
                                                                  
                population[left].Admitance_4.R_Gene.u16 = population[right].Admitance_4.R_Gene.u16;
                population[left].Admitance_4.C_Gene.u16 = population[right].Admitance_4.C_Gene.u16;
                                                                 
                population[left].Admitance_5.R_Gene.u16 = population[right].Admitance_5.R_Gene.u16;
                population[left].Admitance_5.C_Gene.u16 = population[right].Admitance_5.C_Gene.u16;
                                                                   
                population[left].Admitance_6.R_Gene.u16 = population[right].Admitance_6.R_Gene.u16;
                population[left].Admitance_6.C_Gene.u16 = population[right].Admitance_6.C_Gene.u16;


                population[right].fitness = aux;
              
                population[right].Admitance_1.R_Gene.u16 = aux1.R_Gene.u16;
                population[right].Admitance_1.C_Gene.u16 = aux1.C_Gene.u16;
                                              
                population[right].Admitance_2.R_Gene.u16 = aux2.R_Gene.u16;
                population[right].Admitance_2.C_Gene.u16 = aux2.C_Gene.u16;
                                            
                population[right].Admitance_3.R_Gene.u16 = aux3.R_Gene.u16;
                population[right].Admitance_3.C_Gene.u16 = aux3.C_Gene.u16;
                                       
                population[right].Admitance_4.R_Gene.u16 = aux4.R_Gene.u16;
                population[right].Admitance_4.C_Gene.u16 = aux4.C_Gene.u16;
                                               
                population[right].Admitance_5.R_Gene.u16 = aux5.R_Gene.u16;
                population[right].Admitance_5.C_Gene.u16 = aux5.C_Gene.u16;
                                             
                population[right].Admitance_6.R_Gene.u16 = aux6.R_Gene.u16;
                population[right].Admitance_6.C_Gene.u16 = aux6.C_Gene.u16;

            left++;
            right--;

        }

    }while(left <= right);

    if(left_limit < right){
        quickSort(population, left_limit, right);
    }

    if(right_limit > left){
        quickSort(population, left, right_limit);
    }

}

void population_Randomize(FILTER_CHROMOSOME* population) {

    uint32_t temp = 0;

    for(int i = 0; i < NUMBER_OF_MEMEBERS; i++ ){
        population[i].fitness = ((float)(rand() % 1001)) / 1000.0f;

        /* Method with one randomize per gene */
        temp = randomize_32_t();
        population[i].Admitance_1.R_Gene.u16 = (uint16_t)(temp & 0xFFFF);
        population[i].Admitance_1.C_Gene.u16 = (uint16_t)((temp >> 16) & 0xFFFF);

        temp = randomize_32_t();
        population[i].Admitance_2.R_Gene.u16 = (uint16_t)(temp & 0xFFFF);
        population[i].Admitance_2.C_Gene.u16 = (uint16_t)((temp >> 16) & 0xFFFF);
        
        temp = randomize_32_t();
        population[i].Admitance_3.R_Gene.u16 = (uint16_t)(temp & 0xFFFF);
        population[i].Admitance_3.C_Gene.u16 = (uint16_t)((temp >> 16) & 0xFFFF);

        temp = randomize_32_t();
        population[i].Admitance_4.R_Gene.u16 = (uint16_t)(temp & 0xFFFF);
        population[i].Admitance_4.C_Gene.u16 = (uint16_t)((temp >> 16) & 0xFFFF);

        temp = randomize_32_t();
        population[i].Admitance_5.R_Gene.u16 = (uint16_t)(temp & 0xFFFF);
        population[i].Admitance_5.C_Gene.u16 = (uint16_t)((temp >> 16) & 0xFFFF);

        temp = randomize_32_t();
        population[i].Admitance_6.R_Gene.u16 = (uint16_t)(temp & 0xFFFF);
        population[i].Admitance_6.C_Gene.u16 = (uint16_t)((temp >> 16) & 0xFFFF);
      

        #ifdef DEBUG_RANDOMIZE

        print_Genes(population[i], i);

        #endif // DEBUG_RANDOMIZE
    }
}


void print_Bitfield(int value){
    for(int mask = BIT_15_MASK; mask != 0x0000; mask>>=1){
        printf("%i", BIT_VALUE(value,mask) );
    }
}

void print_Genes(FILTER_CHROMOSOME population, int number) {

    printf("Fitness Value for Chromosome %i: %f\n", number, population.fitness);
    printf("Admitance 1:\n");
    printf("Bitfield value for R Gene: ");
    print_Bitfield(population.Admitance_1.R_Gene.u16);
    printf(", Bitfield value for C Gene: ");
    print_Bitfield(population.Admitance_1.C_Gene.u16);
    printf("\n\n");
    printf("Admitance 2:\n");
    printf("Bitfield value for R Gene: ");
    print_Bitfield(population.Admitance_2.R_Gene.u16);
    printf(", Bitfield value for C Gene: ");
    print_Bitfield(population.Admitance_2.C_Gene.u16);
    printf("\n\n");
    printf("Admitance 3:\n");
    printf("Bitfield value for R Gene: ");
    print_Bitfield(population.Admitance_3.R_Gene.u16);
    printf(", Bitfield value for C Gene: ");
    print_Bitfield(population.Admitance_3.C_Gene.u16);
    printf("\n\n");
    printf("Admitance 4:\n");
    printf("Bitfield value for R Gene: ");
    print_Bitfield(population.Admitance_4.R_Gene.u16);
    printf(", Bitfield value for C Gene: ");
    print_Bitfield(population.Admitance_4.C_Gene.u16);
    printf("\n\n");
    printf("Admitance 5:\n");
    printf("Bitfield value for R Gene: ");
    print_Bitfield(population.Admitance_5.R_Gene.u16);
    printf(", Bitfield value for C Gene: ");
    print_Bitfield(population.Admitance_5.C_Gene.u16);
    printf("\n\n");
    printf("Admitance 6:\n");
    printf("Bitfield value for R Gene: ");
    print_Bitfield(population.Admitance_6.R_Gene.u16);
    printf(", Bitfield value for C Gene: ");
    print_Bitfield(population.Admitance_6.C_Gene.u16);
    printf("\n\n");
}

void print_Generation(FILTER_CHROMOSOME *population) {
    for (int i = 0; i < NUMBER_OF_MEMEBERS; i++) {
        print_Genes(population[i], i);
    }
}


void gene_Swap(FILTER_CHROMOSOME *p_population, FILTER_CHROMOSOME *d_population) {

    uint16_t swap_Mask = 0, second_Parent = 0;;

    for (int top_Genes = (NUMBER_OF_MEMEBERS - 1); top_Genes > (NUMBER_OF_MEMEBERS - TOP_GENES - 1); top_Genes--) {   //Best individual of population

        for (int new_Admitance_Gene = 0; new_Admitance_Gene < NUMBER_OF_SONS_PER_TOP_GENE; new_Admitance_Gene++) {
            swap_Mask = VARIABLE_MASK((rand() % (16 - 1) + 1), 1);
            second_Parent = ((rand() % (NUMBER_OF_MEMEBERS - THRESHOLD_TO_RECOMBINE)) + (THRESHOLD_TO_RECOMBINE));

            d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_1.R_Gene.u16 = (p_population[top_Genes].Admitance_1.R_Gene.u16 & swap_Mask) + (p_population[second_Parent].Admitance_1.R_Gene.u16 & ~swap_Mask);
            d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_1.C_Gene.u16 = (p_population[top_Genes].Admitance_1.C_Gene.u16 & swap_Mask) + (p_population[second_Parent].Admitance_1.C_Gene.u16 & ~swap_Mask);

            #ifdef DEBUG_GENE_SWAP
                printf("Number of Descendant: %i\n", ((NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene));
                printf("Admitance 1, ");
                printf("Swap_Mask:");
                print_Bitfield(swap_Mask);
                printf("\n");
                printf("Parent 1 R:");
                print_Bitfield(p_population[top_Genes].Admitance_1.R_Gene.u16);
                printf(", Parent 2 R:");
                print_Bitfield(p_population[second_Parent].Admitance_1.R_Gene.u16);
                printf(", Desecendant R:");
                print_Bitfield(d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_1.R_Gene.u16);
                printf("\n");
                printf("Parent 1 C:");
                print_Bitfield(p_population[top_Genes].Admitance_1.C_Gene.u16);
                printf(", Parent 2 C:");                
                print_Bitfield(p_population[second_Parent].Admitance_1.C_Gene.u16);
                printf(", Desecendant C:");             
                print_Bitfield(d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_1.C_Gene.u16);
                printf("\n\n");
            #endif // DEBUG_GENE_SWAP


            swap_Mask = VARIABLE_MASK((rand() % (16 - 1) + 1), 1);
            second_Parent = ((rand() % (NUMBER_OF_MEMEBERS - THRESHOLD_TO_RECOMBINE)) + (THRESHOLD_TO_RECOMBINE));

            d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_2.R_Gene.u16 = (p_population[top_Genes].Admitance_2.R_Gene.u16 & swap_Mask) + (p_population[second_Parent].Admitance_2.R_Gene.u16 & ~swap_Mask);
            d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_2.C_Gene.u16 = (p_population[top_Genes].Admitance_2.C_Gene.u16 & swap_Mask) + (p_population[second_Parent].Admitance_2.C_Gene.u16 & ~swap_Mask);

            #ifdef DEBUG_GENE_SWAP
                printf("Admitance 2, ");
                printf("Swap_Mask:");
                print_Bitfield(swap_Mask);
                printf("\n");
                printf("Parent 1 :");
                print_Bitfield(p_population[top_Genes].Admitance_2.R_Gene.u16);
                printf(", Parent 2 :");
                print_Bitfield(p_population[second_Parent].Admitance_2.R_Gene.u16);
                printf(", Desecendant :");
                print_Bitfield(d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_2.R_Gene.u16);
                printf("\n");
                printf("Parent 1 C:");
                print_Bitfield(p_population[top_Genes].Admitance_2.C_Gene.u16);
                printf(", Parent 2 C:");              
                print_Bitfield(p_population[second_Parent].Admitance_2.C_Gene.u16);
                printf(", Desecendant C:");           
                print_Bitfield(d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_2.C_Gene.u16);
                printf("\n\n");
            #endif // DEBUG_GENE_SWAP


            swap_Mask = VARIABLE_MASK((rand() % (16 - 1) + 1), 1);
            second_Parent = ((rand() % (NUMBER_OF_MEMEBERS - THRESHOLD_TO_RECOMBINE)) + (THRESHOLD_TO_RECOMBINE));

            d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_3.R_Gene.u16 = (p_population[top_Genes].Admitance_3.R_Gene.u16 & swap_Mask) + (p_population[second_Parent].Admitance_3.R_Gene.u16 & ~swap_Mask);
            d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_3.C_Gene.u16 = (p_population[top_Genes].Admitance_3.C_Gene.u16 & swap_Mask) + (p_population[second_Parent].Admitance_3.C_Gene.u16 & ~swap_Mask);

            #ifdef DEBUG_GENE_SWAP
                printf("Admitance 3, ");
                printf("Swap_Mask:");
                print_Bitfield(swap_Mask);
                printf("\n");
                printf("Parent 1 :");
                print_Bitfield(p_population[top_Genes].Admitance_3.R_Gene.u16);
                printf(", Parent 2 :");
                print_Bitfield(p_population[second_Parent].Admitance_3.R_Gene.u16);
                printf(", Desecendant :");
                print_Bitfield(d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_3.R_Gene.u16);
                printf("\n");
                printf("Parent 1 C:");
                print_Bitfield(p_population[top_Genes].Admitance_3.C_Gene.u16);
                printf(", Parent 2 C:");
                print_Bitfield(p_population[second_Parent].Admitance_3.C_Gene.u16);
                printf(", Desecendant C:");
                print_Bitfield(d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_3.C_Gene.u16);
                printf("\n\n");
            #endif // DEBUG_GENE_SWAP


            swap_Mask = VARIABLE_MASK((rand() % (16 - 1) + 1), 1);
            second_Parent = ((rand() % (NUMBER_OF_MEMEBERS - THRESHOLD_TO_RECOMBINE)) + (THRESHOLD_TO_RECOMBINE));

            d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_4.R_Gene.u16 = (p_population[top_Genes].Admitance_4.R_Gene.u16 & swap_Mask) + (p_population[second_Parent].Admitance_4.R_Gene.u16 & ~swap_Mask);
            d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_4.C_Gene.u16 = (p_population[top_Genes].Admitance_4.C_Gene.u16 & swap_Mask) + (p_population[second_Parent].Admitance_4.C_Gene.u16 & ~swap_Mask);

            #ifdef DEBUG_GENE_SWAP
                printf("Admitance 4, ");
                printf("Swap_Mask:");
                print_Bitfield(swap_Mask);
                printf("\n");
                printf("Parent 1 :");
                print_Bitfield(p_population[top_Genes].Admitance_4.R_Gene.u16);
                printf(", Parent 2 :");
                print_Bitfield(p_population[second_Parent].Admitance_4.R_Gene.u16);
                printf(", Desecendant :");
                print_Bitfield(d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_4.R_Gene.u16);
                printf("\n");
                printf("Parent 1 C:");
                print_Bitfield(p_population[top_Genes].Admitance_4.C_Gene.u16);
                printf(", Parent 2 C:");              
                print_Bitfield(p_population[second_Parent].Admitance_4.C_Gene.u16);
                printf(", Desecendant C:");           
                print_Bitfield(d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_4.C_Gene.u16);
                printf("\n\n");
            #endif // DEBUG_GENE_SWAP

            d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_5.R_Gene.u16 = (p_population[top_Genes].Admitance_5.R_Gene.u16 & swap_Mask) + (p_population[second_Parent].Admitance_5.R_Gene.u16 & ~swap_Mask);
            d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_5.C_Gene.u16 = (p_population[top_Genes].Admitance_5.C_Gene.u16 & swap_Mask) + (p_population[second_Parent].Admitance_5.C_Gene.u16 & ~swap_Mask);

            #ifdef DEBUG_GENE_SWAP
                printf("Admitance 5, ");
                printf("Swap_Mask:");
                print_Bitfield(swap_Mask);
                printf("\n");
                printf("Parent 1 :");
                print_Bitfield(p_population[top_Genes].Admitance_5.R_Gene.u16);
                printf(", Parent 2 :");
                print_Bitfield(p_population[second_Parent].Admitance_5.R_Gene.u16);
                printf(", Desecendant :");
                print_Bitfield(d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_5.R_Gene.u16);
                printf("\n");
                printf("Parent 1 C:");
                print_Bitfield(p_population[top_Genes].Admitance_5.C_Gene.u16);
                printf(", Parent 2 C:");
                print_Bitfield(p_population[second_Parent].Admitance_5.C_Gene.u16);
                printf(", Desecendant C:");
                print_Bitfield(d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_5.C_Gene.u16);
                printf("\n\n");
            #endif // DEBUG_GENE_SWAP

            d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_6.R_Gene.u16 = (p_population[top_Genes].Admitance_6.R_Gene.u16 & swap_Mask) + (p_population[second_Parent].Admitance_6.R_Gene.u16 & ~swap_Mask);
            d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_6.C_Gene.u16 = (p_population[top_Genes].Admitance_6.C_Gene.u16 & swap_Mask) + (p_population[second_Parent].Admitance_6.C_Gene.u16 & ~swap_Mask);

            #ifdef DEBUG_GENE_SWAP
                printf("Admitance 6, ");
                printf("Swap_Mask:");
                print_Bitfield(swap_Mask);
                printf("\n");
                printf("Parent 1 :");
                print_Bitfield(p_population[top_Genes].Admitance_6.R_Gene.u16);
                printf(", Parent 2 :");               
                print_Bitfield(p_population[second_Parent].Admitance_6.R_Gene.u16);
                printf(", Desecendant :");            
                print_Bitfield(d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_6.R_Gene.u16);
                printf("\n");                         
                printf("Parent 1 C:");                
                print_Bitfield(p_population[top_Genes].Admitance_6.C_Gene.u16);
                printf(", Parent 2 C:");              
                print_Bitfield(p_population[second_Parent].Admitance_6.C_Gene.u16);
                printf(", Desecendant C:");           
                print_Bitfield(d_population[(NUMBER_OF_MEMEBERS - top_Genes - 1) * NUMBER_OF_SONS_PER_TOP_GENE + new_Admitance_Gene].Admitance_6.C_Gene.u16);
                printf("\n\n");
            #endif // DEBUG_GENE_SWAP


        }


    }

}

void gene_Mutation(FILTER_CHROMOSOME* population){

    uint16_t temp_mask[6];

    for (int member = 0; member < NUMBER_OF_MEMEBERS; member++) {
        
        for (int i = 0; i < 6; i++) {
            temp_mask[i] = 0x0000;
        }

        for (int mask = BIT_15_MASK; mask != 0x0000; mask >>= 1) {
            if (rand() % 1001 < MUTATION_PROB_R) {
                temp_mask[0] |= mask;
            }
            if (rand() % 1001 < MUTATION_PROB_R) {
                temp_mask[1] |= mask;
            }
            if (rand() % 1001 < MUTATION_PROB_R) {
                temp_mask[2] |= mask;
            }
            if (rand() % 1001 < MUTATION_PROB_R) {
                temp_mask[3] |= mask;
            }
            if (rand() % 1001 < MUTATION_PROB_R) {
                temp_mask[4] |= mask;
            }
            if (rand() % 1001 < MUTATION_PROB_R) {
                temp_mask[5] |= mask;
            }
        }

        #ifdef DEBUG_MUTATION
            printf("Chromosonme %i: \n", member);
            printf("Mutation mask for R member Admitance 1: ");
            print_Bitfield(temp_mask[0]);
            printf("\n");
            printf("Mutation mask for R member Admitance 2: ");
            print_Bitfield(temp_mask[1]);
            printf("\n");
            printf("Mutation mask for R member Admitance 3: ");
            print_Bitfield(temp_mask[2]);
            printf("\n");
            printf("Mutation mask for R member Admitance 4: ");
            print_Bitfield(temp_mask[3]);
            printf("\n");
            printf("Mutation mask for R member Admitance 5: ");
            print_Bitfield(temp_mask[4]);
            printf("\n");
            printf("Mutation mask for R member Admitance 6: ");
            print_Bitfield(temp_mask[5]);
            printf("\n\n");

        #endif // DEBUG_MUTATION

        population[member].Admitance_1.R_Gene.u16 = (population[member].Admitance_1.R_Gene.u16 ^ temp_mask[0]);
        population[member].Admitance_2.R_Gene.u16 = (population[member].Admitance_2.R_Gene.u16 ^ temp_mask[1]);
        population[member].Admitance_3.R_Gene.u16 = (population[member].Admitance_3.R_Gene.u16 ^ temp_mask[2]);
        population[member].Admitance_4.R_Gene.u16 = (population[member].Admitance_4.R_Gene.u16 ^ temp_mask[3]);
        population[member].Admitance_5.R_Gene.u16 = (population[member].Admitance_5.R_Gene.u16 ^ temp_mask[4]);
        population[member].Admitance_6.R_Gene.u16 = (population[member].Admitance_6.R_Gene.u16 ^ temp_mask[5]);

        for (int i = 0; i < 6; i++) {
            temp_mask[i] = 0x0000;
        }

        for (int mask = BIT_15_MASK; mask != 0x0000; mask >>= 1) {
            if (rand() % 1001 < MUTATION_PROB_R) {
                temp_mask[0] |= mask;
            }
            if (rand() % 1001 < MUTATION_PROB_R) {
                temp_mask[1] |= mask;
            }
            if (rand() % 1001 < MUTATION_PROB_R) {
                temp_mask[2] |= mask;
            }
            if (rand() % 1001 < MUTATION_PROB_R) {
                temp_mask[3] |= mask;
            }
            if (rand() % 1001 < MUTATION_PROB_R) {
                temp_mask[4] |= mask;
            }
            if (rand() % 1001 < MUTATION_PROB_R) {
                temp_mask[5] |= mask;
            }
        }

        #ifdef DEBUG_MUTATION
            printf("Chromosonme %i: \n", member);
            printf("Mutation mask for C member Admitance 1: ");
            print_Bitfield(temp_mask[0]);
            printf("\n");
            printf("Mutation mask for C member Admitance 2: ");
            print_Bitfield(temp_mask[1]);
            printf("\n");
            printf("Mutation mask for C member Admitance 3: ");
            print_Bitfield(temp_mask[2]);
            printf("\n");
            printf("Mutation mask for C member Admitance 4: ");
            print_Bitfield(temp_mask[3]);
            printf("\n");
            printf("Mutation mask for C member Admitance 5: ");
            print_Bitfield(temp_mask[4]);
            printf("\n");
            printf("Mutation mask for C member Admitance 6: ");
            print_Bitfield(temp_mask[5]);
            printf("\n\n");

        #endif // DEBUG_MUTATION

        population[member].Admitance_1.C_Gene.u16 = (population[member].Admitance_1.C_Gene.u16 ^ temp_mask[0]);
        population[member].Admitance_2.C_Gene.u16 = (population[member].Admitance_2.C_Gene.u16 ^ temp_mask[1]);
        population[member].Admitance_3.C_Gene.u16 = (population[member].Admitance_3.C_Gene.u16 ^ temp_mask[2]);
        population[member].Admitance_4.C_Gene.u16 = (population[member].Admitance_4.C_Gene.u16 ^ temp_mask[3]);
        population[member].Admitance_5.C_Gene.u16 = (population[member].Admitance_5.C_Gene.u16 ^ temp_mask[4]);
        population[member].Admitance_6.C_Gene.u16 = (population[member].Admitance_6.C_Gene.u16 ^ temp_mask[5]);

    }
}


void filter_Select(const char* filter_Name, int cutoff_1, int cutoff_2, int gain_1, int gain_2){


    if (strcmp(filter_Name, "LPF")) {
        for (int i = 0; i < NUMBER_STEPS; i++) {

            if (i <= cutoff_1) {
                filter_Ideal_Values[i] = gain_1;
            }
            else {
                filter_Ideal_Values[i] = gain_2;
            }
        }    
    }

    else if (strcmp(filter_Name, "HPF")) {
        for (int i = 0; i < NUMBER_STEPS; i++) {

            if (i > cutoff_1) {
                filter_Ideal_Values[i] = gain_1;
            }
            else {
                filter_Ideal_Values[i] = gain_2;
            }
        }
    }

    else if (strcmp(filter_Name, "BPF")) {
        for (int i = 0; i < NUMBER_STEPS; i++) {

            if (i > cutoff_1 && i <=  cutoff_2) {
                filter_Ideal_Values[i] = gain_2;
            }
            else {
                filter_Ideal_Values[i] = gain_1;
            }
        }
    }

    else if (strcmp(filter_Name, "SPF")) {
        for (int i = 0; i < NUMBER_STEPS; i++) {

            if (i <= cutoff_1 | i >= cutoff_2) {
                filter_Ideal_Values[i] = gain_2;
            }
            else {
                filter_Ideal_Values[i] = gain_1;
            }
        }
    }



}

void fitness_Filter_Assign(FILTER_CHROMOSOME population) {




    for (int freq = 0; freq < MAX_FREQUENCY; freq += DELTA_FREQUENCY) {

    }

}


uint32_t randomize_32_t(void) {
    return (uint32_t) ((rand() % ((int)pow(2, (NUMBER_OF_BITS_PER_GENE - 1)))) * (rand() % (int)(pow(2, (NUMBER_OF_BITS_PER_GENE - 1)))));
}