// Physical memory allocator, for user processes,
// kernel stacks, page-table pages,
// and pipe buffers. Allocates whole 4096-byte pages.

#include "types.h"
#include "param.h"
#include "memlayout.h"
#include "spinlock.h"
#include "riscv.h"
#include "defs.h"

void freerange(void *pa_start, void *pa_end);

extern char end[]; // first address after kernel.
                   // defined by kernel.ld.

struct run {
  struct run *next;
};

struct {
  struct spinlock lock;
  struct run *freelist;
} kmem;

#define PA2PGREF_ID(p) (((p)-KERNBASE)/PGSIZE)
#define PGREF_MAX_ENTRIES PA2PGREF_ID(PHYSTOP)

struct page_ref{
  struct spinlock lock; // 每个页面一个锁
  int cnt; // 引用计数
};
struct page_ref page_ref_list[PGREF_MAX_ENTRIES]; // 引用计数数组

void
kinit()
{
  // 初始化每个页面引用计数的锁
  for (int i = 0; i < PGREF_MAX_ENTRIES; ++i) {
    initlock(&(page_ref_list[i].lock), "kpage_ref");
    page_ref_list[i].cnt = 1; 
  }

  initlock(&kmem.lock, "kmem");
  freerange(end, (void*)PHYSTOP);
}

void
freerange(void *pa_start, void *pa_end)
{
  char *p;
  p = (char*)PGROUNDUP((uint64)pa_start);
  for(; p + PGSIZE <= (char*)pa_end; p += PGSIZE)
    kfree(p);
}

// Free the page of physical memory pointed at by v,
// which normally should have been returned by a
// call to kalloc().  (The exception is when
// initializing the allocator; see kinit above.)
void
kfree(void *pa)
{
  struct run *r;

  if(((uint64)pa % PGSIZE) != 0 || (char*)pa < end || (uint64)pa >= PHYSTOP)
    panic("kfree");
    
  // 获取页面锁
  acquire(&((page_ref_list[PA2PGREF_ID((uint64)pa)].lock)));
  // 该页面引用计数减一
  page_ref_list[PA2PGREF_ID((uint64)pa)].cnt--;
  if (page_ref_list[PA2PGREF_ID((uint64)pa)].cnt > 0) {
    // 还有进程在引用该页面，不释放内存，直接返回
    // 释放锁
    release(&((page_ref_list[PA2PGREF_ID((uint64)pa)].lock)));
    return;
  }
  release(&((page_ref_list[PA2PGREF_ID((uint64)pa)].lock)));

  // Fill with junk to catch dangling refs.
  memset(pa, 1, PGSIZE);

  r = (struct run*)pa;

  acquire(&kmem.lock);
  r->next = kmem.freelist;
  kmem.freelist = r;
  release(&kmem.lock);
}

// Allocate one 4096-byte page of physical memory.
// Returns a pointer that the kernel can use.
// Returns 0 if the memory cannot be allocated.
void *
kalloc(void)
{
  struct run *r;

  acquire(&kmem.lock);
  r = kmem.freelist;
  if(r)
    kmem.freelist = r->next;
  release(&kmem.lock);

  if(r)
  {
    memset((char*)r, 5, PGSIZE); // fill with junk
    // 获取页面锁
    acquire(&((page_ref_list[PA2PGREF_ID((uint64)r)].lock)));
    page_ref_list[PA2PGREF_ID((uint64)r)].cnt = 1; // 新页面的引用计数为1
    // 释放锁
    release(&((page_ref_list[PA2PGREF_ID((uint64)r)].lock)));
  }
  return (void*)r;
}

// 引用页面，为 pa 的引用计数增加 1
int krefpage(uint64 pa) {
  if(pa % PGSIZE != 0 // pa位于guard page上 
  || (char*)pa < end // pa位于内核代码区域
  || pa >= PHYSTOP) // pa超过最大内存区域
  { 
    return -1;
  }
  acquire(&((page_ref_list[PA2PGREF_ID(pa)].lock)));
  page_ref_list[PA2PGREF_ID(pa)].cnt++; // 引用计数加一
  release(&((page_ref_list[PA2PGREF_ID(pa)].lock)));
  return 1;
}
// 发生缺页中断时，用于检测该页面地址是否是cow页面
int cow_check(pagetable_t pagetable, uint64 va) {
  if (va > MAXVA) {
    return 0;
  }
  pte_t *pte = walk(pagetable, va, 0);
  if (pte == 0) {
    return 0;
  }
  if (( (*pte) & PTE_V) == 0) {
    return 0;
  }
  return ((*pte) & (PTE_COW));
}

// 当触发缺cow页触发缺页中断时进行实际物理内存分配和映射
uint64
cow_copy(pagetable_t pagetable, uint64 va) 
{
  va = PGROUNDDOWN(va); 
  pte_t *pte = walk(pagetable, va, 0);
  uint64 pa = PTE2PA(*pte);
  // 获取当前页面锁
  acquire(&((page_ref_list[PA2PGREF_ID(pa)].lock)));
  if (page_ref_list[PA2PGREF_ID(pa)].cnt == 1) {
    // 当前页面只有一个进程在使用时，直接返回该页面
    *pte = (*pte) & (~PTE_COW);
    *pte = (*pte) | (PTE_W);
    release(&((page_ref_list[PA2PGREF_ID(pa)].lock)));
    return pa;
  }
  // 一定要释放锁，在kalloc中会申请锁，会导致死锁
  release(&((page_ref_list[PA2PGREF_ID(pa)].lock)));

  // 分配新的内存页面
  uint64 newpa = (uint64)kalloc();
  if (newpa == 0) 
    return 0;

  // 复制旧页面内容到新页面
  memmove((void*)newpa, (void*)pa, PGSIZE);
  *pte = (*pte) & (~PTE_V); // avoid remap 
  uint64 flag = PTE_FLAGS(*pte);
  flag = flag | PTE_W;
  flag = flag & (~PTE_COW);
  // 将申请的物理地址隐射到对应的虚拟地址上
  if(mappages(pagetable, va, PGSIZE, (uint64)newpa, flag) != 0) {
    kfree((void*)newpa);
    return 0;
  }
  // 尝试清除原始的cow副本，此时引用计数可能为0
  kfree((void*)PGROUNDDOWN(pa));
  
  return (uint64)newpa;
}
