/*
	CS 6023 Assignment 3. 
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>

#include <fstream>
#include <iostream>

using namespace std;

using std::cin;
using std::cout;

void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input. 
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	// Input the header information.
	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;
	

	// Input all meshes and store them inside a vector.
	int meshX, meshY ;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ;
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
			}
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ; 
		scenes.push_back (scene) ;
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf (inputFile, "%d", &relations) ;
	int u, v ; 
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	// Input all translations.
	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL; 
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}
	
	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
} 



__global__ void points (int meshid ,int frameSizeX,int frameSizeY,int * outputscene,int *kGlobalCoordinatesX,int *kGlobalCoordinatesY,int V,int *kopmatrix,int r , int c,int *kopacity,int**kmesh){

    		int row = blockIdx.x ;
		int col = threadIdx.x;
int temprow =row+kGlobalCoordinatesX[meshid] ;
int tempcol=col+kGlobalCoordinatesY[meshid];
int id = (row+kGlobalCoordinatesX[meshid]) *frameSizeY +(col+kGlobalCoordinatesY[meshid]) ;

    
    if (id<frameSizeX*frameSizeY&&temprow>=0&&tempcol>=0&&tempcol<frameSizeY&&temprow<frameSizeX){
//  while(atomicCAS(&mutex, 0, 1) != 0);
        if (kopacity[meshid]>kopmatrix[id]){

					atomicExch(&kopmatrix[id],kopacity[meshid]);
					atomicExch(&outputscene[id],kmesh[meshid][row*c+col]);
				}


		}
		
	

}

__global__ void scenemapping(int frameSizeX,int frameSizeY,int * outputscene,int *kGlobalCoordinatesX,int *kGlobalCoordinatesY,int V,int * kopmatrix,int *kFrameSizeX,int *kFrameSizeY,int*kopacity,int **kmesh){
	int id = blockIdx.x * blockDim.x + threadIdx.x ;
	if (id>=V) return ;

    points<<<kFrameSizeX[id],kFrameSizeY[id]>>>(id, frameSizeX, frameSizeY, outputscene,kGlobalCoordinatesX,kGlobalCoordinatesY, V, kopmatrix,kFrameSizeX[id],kFrameSizeY[id],kopacity,kmesh);
}
__global__ void clarity (int *outputscene,int framesizeX){
  int row = threadIdx.x;
  int col = blockIdx.x;
  outputscene[row*framesizeX+col]=0;
}





__global__  void supporter(int *kCsr, int *koffset,int*parentlist,int *accesslist ,int*knumber,int newnum,int vertex) {
 int id = blockIdx.x*blockDim.x + threadIdx.x ;
 if (id<newnum){
    	int loc = atomicAdd(&knumber[0],1);
		parentlist[loc] = vertex ;
		accesslist[loc] = kCsr[koffset[vertex]+id];
 }
} 
__global__ void levelbfs(int *arr,int*kCsr,int *koffset,int number,int *parentlist,int * accesslist, int prev,int *knumber)
{

  int id = blockIdx.x*blockDim.x + threadIdx.x ;
  if (id<number){

    
	 int vertex = accesslist[prev + id] ;
	//  printf("%d  %d %d \n" ,vertex,prev,id);
	 if (vertex!=0){
    // prev --;
	 int parent = parentlist[prev+id] ;
   
	 arr[vertex*2] += arr[parent*2];
	 arr[vertex*2+1] += arr[parent*2+1];
   }
	 int newnum = koffset[vertex+1]-koffset[vertex];
//   int blocks = ceil((float)newnum/1024.0) ;
//   int threads = (number<1024)? number :1024;
  // supporter<<<blocks,threads>>>(kCsr,koffset,parentlist,accesslist,knumber,newnum,vertex);
	 for (int i=0;i<newnum;i++){
		int loc = atomicAdd(&knumber[0],1);
		parentlist[loc] = vertex ;
		accesslist[loc] = kCsr[koffset[vertex]+i];
	 }




  }

}


__host__ void seqbfs(int *arr,int*kCsr,int *koffset,int mesh){
	int newnum = koffset[mesh+1]-koffset[mesh];
	for (int i=0;i<newnum;i++){
		int vertex = kCsr[koffset[mesh]+i];
		arr[vertex*2]+=arr[mesh*2];
		arr[vertex*2+1]+=arr[mesh*2+1];
		seqbfs(arr,kCsr,koffset,vertex);

	}
}

__global__ void adder(int *arr, int*kGlobalCoordinatesX,int *kGlobalCoordinatesY,int V){
	int i = blockIdx.x*blockDim.x +threadIdx.x;
	if (i<V){
		kGlobalCoordinatesX[i]+=arr[i*2];
		kGlobalCoordinatesY[i]+=arr[i*2+1];}
}


int main (int argc, char **argv) {
	
	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;
	int* hFinalPng ; 

	int frameSizeX, frameSizeY ;
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;
	
	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

	// Basic information.
	int V = scenes.size () ;
	int E = edges.size () ;

	int numTranslations = translations.size () ;

	// Convert the scene graph into a csr.
	scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset () ;  
	int *hCsr = scene->get_h_csr () ;
	int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	auto start = std::chrono::high_resolution_clock::now () ;


	// Code begins here.
	// Do not change anything above this comment.

	int *kCsr, *koffset , *kGlobalCoordinatesX , *kGlobalCoordinatesY ;
	cudaMalloc(&kCsr,(E)*sizeof(int));
	cudaMemcpy(kCsr,hCsr,(E)*sizeof(int),cudaMemcpyHostToDevice);
	cudaMalloc(&koffset,(V+1)*sizeof(int));

	cudaMemcpy(koffset,hOffset,(V+1)*sizeof(int),cudaMemcpyHostToDevice);

   vector<int> flatTranslations;
   	

   int *arr = (int*)malloc(V*2*sizeof(int));
memset(arr, 0, V*2*sizeof(int));


   for (int i=0;i<numTranslations;i++){
	
	    if (translations.at(i)[1]==0){
			arr[translations.at(i).at(0)*2]-=translations.at(i).at(2);
		}
		else if (translations.at(i)[1]==1){
			arr[translations.at(i).at(0)*2]+=translations.at(i).at(2);
		}
		else if (translations.at(i)[1]==2){
			arr[translations.at(i).at(0)*2+1]-=translations.at(i).at(2);
		}
		else if (translations.at(i)[1]==3){
			arr[translations.at(i).at(0)*2+1]+=translations.at(i).at(2);
		}

	
   }

   
int *karr ;
cudaMalloc(&karr,2*V*sizeof(int));
cudaMemcpy(karr,arr,2*V*sizeof(int),cudaMemcpyHostToDevice);

	int *parentl , *accessl ;
	cudaMalloc(&parentl,V*sizeof(int));
	cudaMemset(parentl,0,V*sizeof(int));
   	cudaMalloc(&accessl,V*sizeof(int));
	cudaMemset(accessl,0,V*sizeof(int));
int hnumber [1]={0};
int *knumber ;
cudaMalloc(&knumber,sizeof(int));
cudaMemset(knumber,0,sizeof(int));
int prev = 0,knprev= -1 ;

while (prev+1<V){
// printf("%d %d %d \n",hnumber[0],knprev,prev);
	int number = hnumber[0] - knprev;
	int blocks = (number <= 1024) ? 1 : ceil((float)number / 1024.0);
	int threads = (number<=1024) ? number : 1024 ;
  	knprev = hnumber[0] ; 
  
	levelbfs<<<blocks,threads>>>(karr,kCsr,koffset,number,parentl,accessl,prev,knumber);
	// cudaDeviceSynchronize();
	cudaMemcpy(hnumber,knumber,sizeof(int),cudaMemcpyDeviceToHost);

	prev = knprev; 
	// break;
	}
	// seqbfs(arr,hCsr,hOffset,0);





	cudaMalloc(&kGlobalCoordinatesX,(V)*sizeof(int));
	cudaMemcpy(kGlobalCoordinatesX,hGlobalCoordinatesX,(V)*sizeof(int),cudaMemcpyHostToDevice);
	cudaMalloc(&kGlobalCoordinatesY,(V)*sizeof(int));
	cudaMemcpy(kGlobalCoordinatesY,hGlobalCoordinatesY,(V)*sizeof(int),cudaMemcpyHostToDevice);
	// int threadsPerBlock = 1024 ;
	

    int blocksPerGrid= ceil((float)(V)/ 1024.0);
	adder<<< blocksPerGrid,1024>>>(karr,kGlobalCoordinatesX,kGlobalCoordinatesY,V);

	
  cudaDeviceSynchronize();
  	cudaFree(kCsr);
	cudaFree(koffset);

	
  int *outputscene ;
	cudaMalloc(&outputscene,(sizeof (int) * frameSizeX * frameSizeY) ) ;
  clarity<<<frameSizeX,frameSizeY>>>(outputscene,frameSizeX);

	int *kopmatrix ;
	// int threadsPerBlock = 1024; 
	cudaMalloc(&kopmatrix,(sizeof (int) * frameSizeX * frameSizeY) ) ;
   cudaMemset(kopmatrix, 0, sizeof(int) * frameSizeX * frameSizeY);
	// cudaMemcpy(kopmatrix,hopmatrix,(sizeof (int) * frameSizeX * frameSizeY),cudaMemcpyHostToDevice);
	int *kFrameSizeX , *kFrameSizeY,*kopacity ;
	cudaMalloc(&kFrameSizeX,(V)*sizeof(int));
	cudaMalloc(&kFrameSizeY,(V)*sizeof(int));
	
	cudaMalloc(&kopacity,(V)*sizeof(int));
	cudaMemcpy(kopacity,hOpacity,(V)*sizeof(int),cudaMemcpyHostToDevice);

	
	cudaMemcpy(kFrameSizeX,hFrameSizeX,(V)*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(kFrameSizeY,hFrameSizeY,(V)*sizeof(int),cudaMemcpyHostToDevice);

	int **kmesh ;
	cudaMalloc((void**)&kmesh, V * sizeof(int*)); 
	for (int i = 0; i < V; ++i) {
    int* hRow = hMesh[i]; // Get pointer to i-th row of hMesh
    int* kRow;
    cudaMalloc((void**)&kRow, hFrameSizeX[i]*hFrameSizeY[i] * sizeof(int)); // Allocate memory for i-th row on GPU
    cudaMemcpy(kRow, hRow, hFrameSizeX[i]*hFrameSizeY[i]* sizeof(int) , cudaMemcpyHostToDevice); // Copy data from host to device
    cudaMemcpy(&(kmesh[i]), &kRow, sizeof(int*), cudaMemcpyHostToDevice); // Copy pointer to i-th row to GPU
}




	// int numBlocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

	for (int i=0;i<V;i++){
       points<<<hFrameSizeX[i],hFrameSizeY[i]>>>(i, frameSizeX, frameSizeY, outputscene,kGlobalCoordinatesX,kGlobalCoordinatesY, V, kopmatrix,hFrameSizeX[i],hFrameSizeY[i],kopacity,kmesh);
  
 
  }
// scenemapping<<<numBlocks, threadsPerBlock>>>(frameSizeX,frameSizeY,outputscene,kGlobalCoordinatesX,kGlobalCoordinatesY,V,kopmatrix,kFrameSizeX,kFrameSizeY,kopacity,kmesh);
	// cudaDeviceSynchronize();
	cudaFree(kopmatrix);
	cudaFree(kopacity);
	cudaFree(kFrameSizeX);
	cudaFree(kFrameSizeY);
	cudaFree(kmesh);
	cudaFree(kGlobalCoordinatesX);
	cudaFree(kGlobalCoordinatesY);
	cudaFree(koffset);
	


	cudaMemcpy(hFinalPng,outputscene,(sizeof (int) * frameSizeX * frameSizeY),cudaMemcpyDeviceToHost);
	// Do not change anything below this comment.
	// Code ends here.
 
	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;
	cout<<" execution time :"<<timeTaken.count();

	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;	

}
