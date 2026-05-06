#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>

using namespace std;

void sequentialBubbleSort(int *, int);
void parallelBubbleSort(int *, int);
void swap(int &, int &);

void sequentialBubbleSort(int *a, int n)
{
    int swapped;
    for (int i = 0; i < n; i++)
    {
        swapped = 0;
        for (int j = 0; j < n - 1; j++)
        {
            if (a[j] > a[j + 1])
            {
                swap(a[j], a[j + 1]);
                swapped = 1;
            }
        }
        if (!swapped)
            break;
    }
}

void parallelBubbleSort(int *a, int n)
{
    int swapped;
    for (int i = 0; i < n; i++)
    {
        swapped = 0;
        int first = i % 2;

        #pragma omp parallel for shared(a, first)
        for (int j = first; j < n - 1; j += 2)
        {
            if (a[j] > a[j + 1])
            {
                swap(a[j], a[j + 1]);
                swapped = 1;
            }
        }

        if (!swapped)
            break;
    }
}

void swap(int &a, int &b)
{
    int temp = a;
    a = b;
    b = temp;
}

int main()
{
    int *a, n;
    cout << "\nEnter total no of elements: ";
    cin >> n;

    a = new int[n];

    // 🔹 Auto-generate data
    srand(time(0));
    for (int i = 0; i < n; i++)
    {
        a[i] = rand() % 1000; // numbers between 0–999
    }

    cout << "\nGenerated array:\n";
    for (int i = 0; i < n; i++)
        cout << a[i] << " ";

    // Make a copy for fair comparison
    int *b = new int[n];
    for (int i = 0; i < n; i++)
        b[i] = a[i];

    double start_time = omp_get_wtime();
    sequentialBubbleSort(a, n);
    double end_time = omp_get_wtime();

    cout << "\n\nSorted array (Sequential):\n";
    for (int i = 0; i < n; i++)
        cout << a[i] << " ";

    cout << "\nTime taken by sequential: " << end_time - start_time << " seconds\n";

    start_time = omp_get_wtime();
    parallelBubbleSort(b, n);
    end_time = omp_get_wtime();

    cout << "\nSorted array (Parallel):\n";
    for (int i = 0; i < n; i++)
        cout << b[i] << " ";

    cout << "\nTime taken by parallel: " << end_time - start_time << " seconds\n";

    delete[] a;
    delete[] b;

    return 0;
}
