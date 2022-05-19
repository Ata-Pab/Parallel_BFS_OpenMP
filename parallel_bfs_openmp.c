/*
 * Atalay PABUSCU
 * 
 * Multicore Programming Project
 * 
 * Parallelization Of Graph Bredth First Search Algorithm
 * 
 * OZYEGIN UNIVERSITY
 * 
 * With OpenMP
 * Compile: gcc -g -Wall -fopenmp -o parallel_openmp_bfs_project parallel_openmp_bfs_project.c
 * Run: ./parallel_openmp_bfs_project <number_of_threads>
 * Without OpenMP
 * Compile: gcc -g -Wall -o parallel_openmp_bfs_project parallel_openmp_bfs_project.c
 * Run: ./parallel_openmp_bfs_project
 * 
 * 
 * Pseudocode of Sequential BFS Algorithm:
 * 
 *   create a queue Q 
 *   mark v as visited and put v into Q 
 *   while Q is non-empty 
 *     remove the head u of Q 
 *     mark and enqueue all (unvisited) neighbours of u
 *
 *
 * queue = new Queue();
 * queue.enqueue(r);  // initialize the queue to contain only the root vertex r
 * distance of r = 0;
 * while (!queue.isEmpty()) {
 *  x = queue.dequeue(); {  // remove vertex x from the queue
 *   for (each vertex y that is adjacent to x) {
 *     if (y has not been visited yet) {
 *       y's distance = x's distance + 1;
 *       y's back-pointer = x;
 *       queue.enqueue(y);   // insert y into the queue
 *     }
 *   }
 * }
 * 
 * Reference: Direction-Optimizing Breadth First Search, Scott Beamer, Krste Asanovic, David Patterson, Electrical Engineering and Computer Science Department University of California, Berkeley
 * https://parlab.eecs.berkeley.edu/sites/all/parlab/files/main.pdf
 * https://www.programiz.com/dsa/graph-bfs
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define FALSE 0
#define TRUE  1
#define NONE -1

#define GRAPH1 10
#define GRAPH2 11
#define GRAPH3 12

#define PARALLEL_OPENMP TRUE
#define CUSTOM_GRAPHS GRAPH3

#if (PARALLEL_OPENMP == TRUE)    // Compiler switch for BFS Parallelization using OPENMP
#include <omp.h>
#endif

#define GET_TIME(now) { \
   struct timeval t; \
   gettimeofday(&t, NULL); \
   now = t.tv_sec + t.tv_usec/1000000.0; \
}

#if(CUSTOM_GRAPHS == GRAPH1)
#define QUEUE_SIZE 8
#define NUMBER_OF_NODES 8
#elif(CUSTOM_GRAPHS == GRAPH2)
#define QUEUE_SIZE 14
#define NUMBER_OF_NODES 14
#elif(CUSTOM_GRAPHS == GRAPH3)
#define QUEUE_SIZE 16
#define NUMBER_OF_NODES 16
#else
#define QUEUE_SIZE 2000
#define NUMBER_OF_NODES 2000
#endif

int thread_count = 1;   // Default thread number (No parallelization-just main thread)
int total_nodes = NUMBER_OF_NODES;   // Numbor of nodes in Graph

typedef struct queue {
  int items[QUEUE_SIZE];
  int front;
  int rear;
}queue;

struct node {
  int vertex;
  struct node* next;
};

struct node* app_CreateNode(int);

struct Graph {
  int numVertices;
  struct node** adjLists;
  int* visited;
};

queue* app_CreateQueue();
void app_Enqueue(queue* q, int);
int app_Dequeue(queue* q);
int app_IsQueueEmpty(queue* q);
int app_IsThereAnyNode(int vertex, queue* q);
void app_TopDownStep(struct Graph* graph, queue* frontier, queue* next, queue* parents);
void app_PrintGraphTraversal(queue* traversal);
void app_CreateCustomGraph(struct Graph* graph);
int app_GetUniqueRnd(int node, int cnt);
void app_CreateRndGraph(struct Graph* graph, int node_num);


/*
 * function top-down-step(frontier, next, parents)
 *  for v ∈ frontier do
 *    for n ∈ neighbors[v] do
 *      if parents[n] = -1 then
 *          parents[n] ← v
 *          next ← next ∪ {n}
 *      end if
 *    end for
 *  end for
*/
void app_TopDownStep(struct Graph* graph, queue* frontier, queue* next, queue* parents)
{
  # if (PARALLEL_OPENMP == TRUE)
  # pragma omp prallel num_threads(thread_count) default(none) shared(total_nodes, frontier, next, parents) private(adjNode, adjVertex, vertex, node, thread_num) schedule(dynamic, 1)
  # endif
  for(int vertex = 0; vertex < total_nodes; vertex++)
  {
    if (app_IsThereAnyNode(vertex, frontier) == TRUE)   // for v ∈ frontier do
    {
      //printf("%d", vertex);
      struct node* adjNode = graph->adjLists[vertex];   // for n ∈ adjacentList[vertex]
      while(adjNode)
      {
        int adjVertex = adjNode->vertex;
  #     if (PARALLEL_OPENMP == TRUE)
        #pragma omp parallel for
  #     endif
        for(int node = 0; node < total_nodes; node++)
        {
          if (node == adjVertex)   // for n ∈ adjacentList[vertex] do
          {
            if(parents->items[node] == NONE && node != 0)   // if parents[n] = -1 then
            {
  #           if (PARALLEL_OPENMP == TRUE)
              int thread_num = omp_get_thread_num();
  #           else
              int thread_num = 1;
  #           endif
              printf("\nAdjacent node of %d vertex (OpenMP Thread No: %d): %d\n",vertex, adjVertex, thread_num);
              printf("Parent of %d adjacent vertex (OpenMP Thread No: %d): %d\n", adjVertex, vertex, thread_num);
              parents->items[node] = vertex;   // parents[n] ← v
              printf("Add node %d to next queue (OpenMP Thread No: %d)", node, thread_num);
              //next->items[node] = node;
              app_Enqueue(next, node);   // next ← next U {n}
            }
          }
        }
        adjNode = adjNode->next;    // Traverse all adjacent nodes of this vertex
      }
    }
  }
}

/*
 * BFS ALGORITHM
 * function breadth-first-search(graph, source)
 *  frontier ← {source}
 *  next ← {}
 *  parents ← [-1,-1,. . . -1]
 *  while frontier 6= {} do
 *    top-down-step(frontier, next, parents)
 *    frontier ← next
 *    next ← {}
 *  end while
 *  return tree
*/
void app_BFS(struct Graph* graph, int startVertex) {
  queue* frontier = app_CreateQueue();  // frontier ← {}
  queue* next = app_CreateQueue();      // next ← {}
  queue* parents = app_CreateQueue();   // parents ← {}
  queue* traversal_result = app_CreateQueue();

  graph->visited[startVertex] = 1;
  app_Enqueue(frontier, startVertex);   // frontier ← {source}

  while (!app_IsQueueEmpty(frontier))   // while frontier != {} do: If frontier is not empty
  {
    app_TopDownStep(graph, frontier, next, parents);

    int currentVertex = app_Dequeue(frontier);
    app_Enqueue(traversal_result, currentVertex);
    
    while(!app_IsQueueEmpty(next))
    {
      // frontier ← next
      int currentVertex = app_Dequeue(next);
      app_Enqueue(frontier, currentVertex);
    }

    next = app_CreateQueue();   // next ← {}
  }

  app_PrintGraphTraversal(traversal_result);
}

// Creating a node
struct node* app_CreateNode(int v) {
  struct node* newNode = malloc(sizeof(struct node));
  newNode->vertex = v;
  newNode->next = NULL;
  return newNode;
}

void app_PrintGraphTraversal(queue* traversal)
{
  printf("\n\nBFS Queue Traversal:\n");
  while (!app_IsQueueEmpty(traversal))
    printf("Visited Vertex: %d\n", app_Dequeue(traversal));
}

int app_IsThereAnyNode(int vertex, queue* q)
{
  for(int node = 0; node < QUEUE_SIZE; node++)
  {
    if(vertex == q->items[node])
    {
      return TRUE;
    }
  }
  return FALSE;
}

// Creating a Graph
struct Graph* app_CreateGraph(int vertices) {
  struct Graph* graph = malloc(sizeof(struct Graph));
  graph->numVertices = vertices;

  graph->adjLists = malloc(vertices * sizeof(struct node*));
  graph->visited = malloc(vertices * sizeof(int));

  int i;
  for (i = 0; i < vertices; i++) {
    graph->adjLists[i] = NULL;
    graph->visited[i] = 0;
  }

  return graph;
}

// Add Edge to a Graph
void app_AddEdgeToGraph(struct Graph* graph, int src, int dest) {
  // Add edge from src to dest
  struct node* newNode = app_CreateNode(dest);
  newNode->next = graph->adjLists[src];
  graph->adjLists[src] = newNode;

  // Add edge from dest to src
  newNode = app_CreateNode(src);
  newNode->next = graph->adjLists[dest];
  graph->adjLists[dest] = newNode;
}

// Create a queue
queue* app_CreateQueue() {
  queue* q = malloc(sizeof(queue));
  q->front = NONE;
  q->rear = NONE;
  for (int node = 0; node < NUMBER_OF_NODES; node++)
  {
    q->items[node] = NONE;
  }
  return q;
}

// Check if the queue is empty
int app_IsQueueEmpty(queue* q) {
  if (q->rear == NONE)
    return 1;
  else
    return 0;
}

// Adding elements into queue
void app_Enqueue(queue* q, int value) {
  if (q->rear == QUEUE_SIZE - 1)
  {
    //printf("\nQueue is Full!!");
  }
  else {
    if (q->front == NONE)
      q->front = 0;
    q->rear++;
    q->items[q->rear] = value;
  }
}

// Removing elements from queue
int app_Dequeue(queue* q) {
  int item;
  if (app_IsQueueEmpty(q)) {
    //printf("Queue is empty");
    item = NONE;
  } else {
    item = q->items[q->front];
    q->front++;
    if (q->front > q->rear) {
      //printf("Resetting queue ");
      q->front = q->rear = NONE;
    }
  }
  return item;
}

void app_CreateRndGraph(struct Graph* graph, int node_num)
{
  srand(time(NULL));
  // When you do srand(<number>) you select the book rand() will use from that point forward.
  // If you don't select a book, the rand() function takes numbers from book #1 (same as srand(1)).
  // It always returns same value 

  for(int node = 0; node < node_num; node++)
  {
    int rnd_node = app_GetUniqueRnd(node, node_num);
    app_AddEdgeToGraph(graph, node, rnd_node);
    printf("\nNode.%d  =>  Rnd_Dest_Node: %d", node, rnd_node);
  }
  printf("\n\n");
}

int app_GetUniqueRnd(int node, int cnt)
{
  int val = rand() % cnt;

  if(node != val)
  {
    return val;
  }
  else
    app_GetUniqueRnd(node, cnt);
}

void app_CreateCustomGraph(struct Graph* graph)
{
  #if(CUSTOM_GRAPHS == GRAPH1)
  app_AddEdgeToGraph(graph, 0, 4);
  app_AddEdgeToGraph(graph, 0, 5);

  app_AddEdgeToGraph(graph, 1, 5);

  app_AddEdgeToGraph(graph, 2, 3);
  app_AddEdgeToGraph(graph, 2, 4);
  app_AddEdgeToGraph(graph, 2, 6);

  app_AddEdgeToGraph(graph, 3, 2);
  app_AddEdgeToGraph(graph, 3, 7);

  app_AddEdgeToGraph(graph, 4, 0);
  app_AddEdgeToGraph(graph, 4, 2);

  app_AddEdgeToGraph(graph, 5, 0);
  app_AddEdgeToGraph(graph, 5, 1);
  app_AddEdgeToGraph(graph, 5, 7);

  app_AddEdgeToGraph(graph, 6, 2);
  
  app_AddEdgeToGraph(graph, 7, 3);
  app_AddEdgeToGraph(graph, 7, 5);
  #elif(CUSTOM_GRAPHS == GRAPH2)
  app_AddEdgeToGraph(graph, 0, 4);
  app_AddEdgeToGraph(graph, 0, 5);

  app_AddEdgeToGraph(graph, 1, 5);
  app_AddEdgeToGraph(graph, 1, 8);

  app_AddEdgeToGraph(graph, 2, 3);
  app_AddEdgeToGraph(graph, 2, 4);
  app_AddEdgeToGraph(graph, 2, 6);

  app_AddEdgeToGraph(graph, 3, 2);
  app_AddEdgeToGraph(graph, 3, 7);

  app_AddEdgeToGraph(graph, 4, 0);
  app_AddEdgeToGraph(graph, 4, 2);
  app_AddEdgeToGraph(graph, 4, 9);

  app_AddEdgeToGraph(graph, 5, 0);
  app_AddEdgeToGraph(graph, 5, 1);
  app_AddEdgeToGraph(graph, 5, 7);

  app_AddEdgeToGraph(graph, 6, 2);
  app_AddEdgeToGraph(graph, 6, 11);
  
  app_AddEdgeToGraph(graph, 7, 3);
  app_AddEdgeToGraph(graph, 7, 5);
  app_AddEdgeToGraph(graph, 7, 11);
  app_AddEdgeToGraph(graph, 7, 12);
  app_AddEdgeToGraph(graph, 7, 13);
  
  app_AddEdgeToGraph(graph, 8, 1);
  app_AddEdgeToGraph(graph, 8, 10);
  
  app_AddEdgeToGraph(graph, 9, 4);
  app_AddEdgeToGraph(graph, 9, 13);
  
  app_AddEdgeToGraph(graph, 10, 8);
  
  app_AddEdgeToGraph(graph, 11, 6);
  app_AddEdgeToGraph(graph, 11, 7);
  
  app_AddEdgeToGraph(graph, 12, 7);
  
  app_AddEdgeToGraph(graph, 13, 9);
  app_AddEdgeToGraph(graph, 13, 7);
  #elif(CUSTOM_GRAPHS == GRAPH3)
  app_AddEdgeToGraph(graph, 0, 1);
  app_AddEdgeToGraph(graph, 0, 2);

  app_AddEdgeToGraph(graph, 1, 4);
  app_AddEdgeToGraph(graph, 1, 0);

  app_AddEdgeToGraph(graph, 2, 5);

  app_AddEdgeToGraph(graph, 3, 12);
  app_AddEdgeToGraph(graph, 3, 4);

  app_AddEdgeToGraph(graph, 4, 6);
  app_AddEdgeToGraph(graph, 4, 3);

  app_AddEdgeToGraph(graph, 5, 8);
  app_AddEdgeToGraph(graph, 5, 12);

  app_AddEdgeToGraph(graph, 6, 7);
  app_AddEdgeToGraph(graph, 6, 4);
  
  app_AddEdgeToGraph(graph, 7, 6);
  app_AddEdgeToGraph(graph, 7, 11);
  app_AddEdgeToGraph(graph, 7, 12);
  
  app_AddEdgeToGraph(graph, 8, 9);
  
  app_AddEdgeToGraph(graph, 9, 10);
  
  app_AddEdgeToGraph(graph, 10, 15);
  app_AddEdgeToGraph(graph, 10, 13);
  
  app_AddEdgeToGraph(graph, 11, 7);
  
  app_AddEdgeToGraph(graph, 12, 7);
  app_AddEdgeToGraph(graph, 12, 13);
  app_AddEdgeToGraph(graph, 12, 14);
  
  app_AddEdgeToGraph(graph, 13, 10);
  app_AddEdgeToGraph(graph, 13, 12);
  
  app_AddEdgeToGraph(graph, 14, 12);
  
  app_AddEdgeToGraph(graph, 15, 10);
  #else

  #endif
}

int main(int argc, char* argv[]) {
  double t_start, t_finish;
  struct Graph* graph = app_CreateGraph(NUMBER_OF_NODES);
  thread_count = strtol(argv[1], NULL, 10);

  #if(CUSTOM_GRAPHS == FALSE)
  app_CreateRndGraph(graph, NUMBER_OF_NODES);
  #else
  app_CreateCustomGraph(graph);
  #endif

  GET_TIME(t_start);
  app_BFS(graph, 0);
  GET_TIME(t_finish);

  printf("BFS elapsed time = %e seconds\n", t_finish - t_start);

  return 0;
}