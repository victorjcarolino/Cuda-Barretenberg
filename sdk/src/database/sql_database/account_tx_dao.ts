import { AccountId, AliasHash } from '@aztec/barretenberg/account_id';
import { TxId } from '@aztec/barretenberg/tx_id';
import { AfterInsert, AfterLoad, AfterUpdate, Column, Entity, Index, PrimaryColumn } from 'typeorm';
import { accountIdTransformer, aliasHashTransformer, txIdTransformer } from './transformer';

@Entity({ name: 'accountTx' })
export class AccountTxDao {
  @PrimaryColumn('blob', { transformer: [txIdTransformer] })
  public txId!: TxId;

  @Index({ unique: false })
  @Column('blob', { transformer: [accountIdTransformer] })
  public userId!: AccountId;

  @Column('blob', { nullable: true, transformer: [aliasHashTransformer] })
  public aliasHash!: AliasHash;

  @Column({ nullable: true })
  public newSigningPubKey1?: Buffer;

  @Column({ nullable: true })
  public newSigningPubKey2?: Buffer;

  @Column()
  public migrated!: boolean;

  @Column()
  public txRefNo!: number;

  @Column()
  public created!: Date;

  @Column({ nullable: true })
  public settled?: Date;

  @AfterLoad()
  @AfterInsert()
  @AfterUpdate()
  afterLoad() {
    if (this.settled === null) {
      delete this.settled;
    }
  }
}