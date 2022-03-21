import { promises as fs } from 'fs';
import { WorldStateDb, PutEntry } from '../../world_state_db';
import { toBigIntBE, toBufferBE } from '../../bigint_buffer';
import * as pathTools from 'path';
import { getInitData } from './init_config';

const NOTE_LENGTH = 32;
const ADDRESS_LENGTH = 64;
const ALIAS_HASH_LENGTH = 28;
const NULLIFIER_LENGTH = 32;
const SIGNING_KEY_LENGTH = 32;

export interface AccountNotePair {
  note1: Buffer;
  note2: Buffer;
}

export interface AccountAlias {
  nonce: number;
  aliasHash: Buffer;
  address: Buffer;
}

export interface SigningKeys {
  signingKey1: Buffer;
  signingKey2: Buffer;
}

export interface Roots {
  dataRoot: Buffer;
  nullRoot: Buffer;
  rootsRoot: Buffer;
}

export interface AccountData {
  notes: AccountNotePair;
  nullifier: Buffer;
  alias: AccountAlias;
  signingKeys: SigningKeys;
}

export class InitHelpers {
  public static getInitRoots(chainId: number) {
    const { initDataRoot, initNullRoot, initRootsRoot } = getInitData(chainId).initRoots;
    return {
      initDataRoot: Buffer.from(initDataRoot, 'hex'),
      initNullRoot: Buffer.from(initNullRoot, 'hex'),
      initRootsRoot: Buffer.from(initRootsRoot, 'hex'),
    };
  }

  public static getInitDataSize(chainId: number) {
    return getInitData(chainId).initDataSize;
  }

  public static getAccountDataFile(chainId: number) {
    if (!getInitData(chainId).accounts) {
      return undefined;
    }
    const relPathToFile = getInitData(chainId).accounts;
    const fullPath = pathTools.resolve(__dirname, relPathToFile);
    return fullPath;
  }

  public static getRootDataFile(chainId: number) {
    if (!getInitData(chainId).roots) {
      return undefined;
    }
    const relPathToFile = getInitData(chainId).roots;
    const fullPath = pathTools.resolve(__dirname, relPathToFile);
    return fullPath;
  }

  public static async writeData(filePath: string, data: Buffer) {
    const path = pathTools.resolve(__dirname, filePath);
    const fileHandle = await fs.open(path, 'w');
    const { bytesWritten } = await fileHandle.write(data);
    await fileHandle.close();
    return bytesWritten;
  }

  public static async readData(filePath: string) {
    const path = pathTools.resolve(__dirname, filePath);
    try {
      const fileHandle = await fs.open(path, 'r');
      const data = await fileHandle.readFile();
      await fileHandle.close();
      return data;
    } catch (err) {
      console.log(`Failed to read file: ${path}. Error: ${err}`);
      return Buffer.alloc(0);
    }
  }

  public static async writeAccountTreeData(accountData: AccountData[], filePath: string) {
    accountData.forEach(account => {
      if (account.notes.note1.length !== NOTE_LENGTH) {
        throw new Error(`Note1 has length ${account.notes.note1.length}, it should be ${NOTE_LENGTH}`);
      }
      if (account.notes.note2.length !== NOTE_LENGTH) {
        throw new Error(`Note2 has length ${account.notes.note2.length}, it should be ${NOTE_LENGTH}`);
      }
      if (account.alias.aliasHash.length !== ALIAS_HASH_LENGTH) {
        throw new Error(`Alias hash has length ${account.alias.aliasHash.length}, it should be ${ALIAS_HASH_LENGTH}`);
      }
      if (account.alias.address.length !== ADDRESS_LENGTH) {
        throw new Error(
          `Alias grumpkin address has length ${account.alias.address.length}, it should be ${ADDRESS_LENGTH}`,
        );
      }
      if (account.nullifier.length !== NULLIFIER_LENGTH) {
        throw new Error(`Nullifier has length ${account.nullifier.length}, it should be ${NULLIFIER_LENGTH}`);
      }
      if (account.signingKeys.signingKey1.length !== SIGNING_KEY_LENGTH) {
        throw new Error(
          `Signing Key 1 has length ${account.signingKeys.signingKey1.length}, it should be ${SIGNING_KEY_LENGTH}`,
        );
      }
      if (account.signingKeys.signingKey2.length !== SIGNING_KEY_LENGTH) {
        throw new Error(
          `Signing Key 2 has length ${account.signingKeys.signingKey2.length}, it should be ${SIGNING_KEY_LENGTH}`,
        );
      }
    });
    const dataToWrite = accountData.flatMap(account => {
      const nonBuf = Buffer.alloc(4);
      nonBuf.writeUInt32BE(account.alias.nonce);
      return [
        nonBuf,
        account.alias.aliasHash,
        account.alias.address,
        account.notes.note1,
        account.notes.note2,
        account.nullifier,
        account.signingKeys.signingKey1,
        account.signingKeys.signingKey2,
      ];
    });
    return await this.writeData(filePath, Buffer.concat(dataToWrite));
  }

  public static parseAccountTreeData(data: Buffer) {
    const lengthOfAccountData =
      4 + ALIAS_HASH_LENGTH + ADDRESS_LENGTH + 2 * NOTE_LENGTH + NULLIFIER_LENGTH + 2 * SIGNING_KEY_LENGTH;
    const numAccounts = data.length / lengthOfAccountData;
    if (numAccounts === 0) {
      return [];
    }
    const accounts = new Array<AccountData>(numAccounts);
    for (let i = 0; i < numAccounts; i++) {
      let start = i * lengthOfAccountData;
      const alias: AccountAlias = {
        nonce: data.readUInt32BE(start),
        aliasHash: data.slice(start + 4, start + (4 + ALIAS_HASH_LENGTH)),
        address: data.slice(start + (4 + ALIAS_HASH_LENGTH), start + (4 + ALIAS_HASH_LENGTH + ADDRESS_LENGTH)),
      };
      start += 4 + ALIAS_HASH_LENGTH + ADDRESS_LENGTH;
      const notes: AccountNotePair = {
        note1: data.slice(start, start + NOTE_LENGTH),
        note2: data.slice(start + NOTE_LENGTH, start + 2 * NOTE_LENGTH),
      };
      start += 2 * NOTE_LENGTH;
      const nullifier = data.slice(start, start + NULLIFIER_LENGTH);
      start += NULLIFIER_LENGTH;
      const signingKeys: SigningKeys = {
        signingKey1: data.slice(start, start + SIGNING_KEY_LENGTH),
        signingKey2: data.slice(start + SIGNING_KEY_LENGTH, start + 2 * SIGNING_KEY_LENGTH),
      };
      const account: AccountData = {
        notes,
        nullifier,
        alias,
        signingKeys,
      };
      accounts[i] = account;
    }
    return accounts;
  }

  public static async readAccountTreeData(filePath: string) {
    const data = await this.readData(filePath);
    return this.parseAccountTreeData(data);
  }

  public static async populateDataAndRootsTrees(
    accounts: AccountData[],
    merkleTree: WorldStateDb,
    dataTreeIndex: number,
    rootsTreeIndex: number,
  ) {
    const stepSize = 1000;
    const entries = accounts.flatMap((account, index): PutEntry[] => {
      return [
        {
          treeId: dataTreeIndex,
          index: BigInt(index * 2),
          value: account.notes.note1,
        },
        {
          treeId: dataTreeIndex,
          index: BigInt(1 + index * 2),
          value: account.notes.note2,
        },
      ];
    });
    let i = 0;
    while (i < entries.length) {
      if (i % 1000 == 0) {
        console.log(`Inserted ${i}/${entries.length} entries into data tree...`);
      }
      await merkleTree.batchPut(entries.slice(i, i + stepSize));
      i += stepSize;
    }
    const dataRoot = merkleTree.getRoot(dataTreeIndex);
    await merkleTree.put(rootsTreeIndex, BigInt(0), dataRoot);
    const rootsRoot = merkleTree.getRoot(rootsTreeIndex);
    return { dataRoot, rootsRoot };
  }

  public static async populateNullifierTree(accounts: AccountData[], merkleTree: WorldStateDb, nullTreeIndex: number) {
    const stepSize = 1000;
    const emptyBuffer = Buffer.alloc(32, 0);
    const entries = accounts.flatMap((account): PutEntry[] => {
      const nullifiers: Array<PutEntry> = [];
      if (account.nullifier.compare(emptyBuffer)) {
        nullifiers.push({
          treeId: nullTreeIndex,
          index: toBigIntBE(account.nullifier),
          value: toBufferBE(BigInt(1), 32),
        });
      }
      return nullifiers;
    });
    let i = 0;
    while (i < entries.length) {
      if (i % 1000 == 0) {
        console.log(`Inserted ${i}/${entries.length} entries into nullifier tree...`);
      }
      await merkleTree.batchPut(entries.slice(i, i + stepSize));
      i += stepSize;
    }
    const root = merkleTree.getRoot(nullTreeIndex);
    return root;
  }

  public static async writeRoots(roots: Roots, filePath: string) {
    return await this.writeData(
      filePath,
      Buffer.from(
        JSON.stringify({
          dataRoot: roots.dataRoot.toString('hex'),
          nullRoot: roots.nullRoot.toString('hex'),
          rootsRoot: roots.rootsRoot.toString('hex'),
        }),
      ),
    );
  }

  public static async readRoots(filePath: string) {
    const data = await this.readData(filePath);
    if (!data.length) {
      return;
    }
    const roots = JSON.parse(data.toString());
    return {
      dataRoot: Buffer.from(roots.dataRoot, 'hex'),
      nullRoot: Buffer.from(roots.nullRoot, 'hex'),
      rootsRoot: Buffer.from(roots.rootsRoot, 'hex'),
    };
  }
}
