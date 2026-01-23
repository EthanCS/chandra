use keter::prelude::Value as _; // Shadowed for some reason
use keter::prelude::*;
use yesod::rand::pcg;

pub const INVALID_KEY: u32 = u32::MAX;

#[tracked]
fn hash32(key: Expr<u32>) -> Expr<u32> {
    pcg(key)
}

pub type Key = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Value)]
#[repr(C)]
pub struct Value {
    index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Value)]
#[repr(C)]
pub struct Entry {
    key: Key,
    value: Value,
}
impl Entry {
    pub const EMPTY: Self = Self {
        key: INVALID_KEY,
        value: Value { index: u32::MAX },
    };
}

#[derive(Debug)]
pub struct HashTable {
    pub num_tables: u32,
    capacity: u32,
    // per entry
    entries: Buffer<Entry>,
    // per datum
    freelist: Buffer<u32>,
    data_key: Buffer<Key>,
    // per table
    table_data_start: Buffer<u32>,
    freelist_index: Buffer<u32>, // reset to table_data_start after each build
}

/*
update:
- reset freelist_index to table_data_start
- clear hashmap entries
- call build_freelists (repopulates hashmap and freelist)
- reset freelist_index to table_data_start (also want to check the max just in case?)
- perform insertions
- perform queries
*/
impl HashTable {
    pub fn new(
        num_tables: u32,
        entry_capacity: u32,
        data_capacities: impl Iterator<Item = u32>,
    ) -> Self {
        let total_entry_capacity = entry_capacity * num_tables;
        let mut table_data_start = Vec::new();
        let mut total_data_capacity = 0;
        for cap in data_capacities {
            table_data_start.push(total_data_capacity);
            total_data_capacity += cap;
        }
        assert_eq!(table_data_start.len() as u32, num_tables);
        Self {
            num_tables,
            capacity: entry_capacity,

            entries: DEVICE.create_buffer_from_fn(total_entry_capacity as usize, |_| Entry::EMPTY),
            freelist: DEVICE.create_buffer(total_data_capacity as usize),
            data_key: DEVICE.create_buffer_from_fn(total_data_capacity as usize, |_| INVALID_KEY),

            table_data_start: DEVICE.create_buffer_from_slice(&table_data_start),
            freelist_index: DEVICE.create_buffer(num_tables as usize),
        }
    }
    #[tracked]
    fn insert_entry_index(&self, key: Expr<u32>, table: Expr<u32>, index: Expr<u32>) {
        let entry_index = (hash32(key) % self.capacity).var();
        let offset = self.capacity * table;
        loop {
            let entry = self.entries.atomic_ref(entry_index + offset);
            let prev = entry.key.compare_exchange(INVALID_KEY, key);
            if prev == key {
                // already inserted, do nothing
                break;
            } else if prev == INVALID_KEY {
                entry.value.index.exchange(index);
                break;
            }

            *entry_index = (entry_index + 1) % self.capacity;
        }
    }
    #[tracked]
    pub fn insert(&self, key: Expr<u32>, table: Expr<u32>) {
        let entry_index = (hash32(key) % self.capacity).var();
        let offset = self.capacity * table;
        loop {
            let entry = self.entries.atomic_ref(offset + entry_index);
            let prev = entry.key.compare_exchange(INVALID_KEY, key);
            if prev == key {
                // already inserted, do nothing
                break;
            } else if prev == INVALID_KEY {
                let index = self
                    .freelist
                    .read(self.freelist_index.atomic_ref(table).fetch_add(1));
                entry.value.index.exchange(index);
                self.data_key.write(index, key);
                break;
            }

            *entry_index = (entry_index + 1) % self.capacity;
        }
    }
    #[tracked]
    pub fn get(&self, key: Expr<u32>, table: Expr<u32>) -> Expr<u32> {
        let entry_index = (hash32(key) % self.capacity).var();
        let offset = self.capacity * table;
        let result = u32::MAX.var();
        loop {
            let entry = self.entries.read(offset + entry_index);
            if entry.key == key {
                *result = entry.value.index;
                break;
            }
        }
        **result
    }
    #[tracked]
    pub fn build_freelists(&self, remove: impl Fn(Expr<u32>) -> Expr<bool>) {
        let data_index = dispatch_id().x;
        let table = 0_u32.var();
        // TODO: Can be optimized if we know more about the table sizes
        for i in 1..self.num_tables {
            let t = self.table_data_start.read(i);
            if data_index >= t {
                *table = i;
            } else {
                break;
            }
        }
        let key = self.data_key.read(data_index);
        let occupied = (key != INVALID_KEY).var();
        // Separate?
        if occupied {
            if remove(data_index) {
                self.data_key.write(data_index, INVALID_KEY);
                *occupied = false;
            } else {
                let key = self.data_key.read(data_index);
                self.insert_entry_index(key, **table, data_index);
            }
        }
        if !occupied {
            let freelist_index = self.freelist_index.atomic_ref(table).fetch_add(1);
            self.freelist.write(freelist_index, data_index);
        }
    }
}
