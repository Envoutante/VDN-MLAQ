# -*- coding: utf-8 -*-
"""Per-case Q comparison using production compute/get_best functions.

Primary values and actions are reused; only the counterpart is computed on a
private copy. Q_MLAQ = gamma * child.value is an exported comparison scale,
not a replacement for MLAQ get_best. Equality with scaled VDN is not assumed.
"""
import os
import copy

GAMMA = 0.995
TOL_TIE = 1e-9          # 「并列最优」的容差：不能按精确相等判，
                        # 否则 1e-11 的迭代误差会把 VDN 的平局组拆散

HDR = ['snapshot', 'phase', '触发点', '依据', '状态ID', '当前状态',
       '联合动作', 'Alice动作', 'Bob动作', 'Chad动作', '下一状态',
       'Q_MLAQ', 'Q_VDN', 'Q_Alice', 'Q_Bob', 'Q_Chad', 'Q_VDN×γ', 'Δ(VDN×γ-MLAQ)',
       'UCB_MLAQ', 'UCB_VDN', '是虚拟节点', 'MLAQ并列最优', 'VDN并列最优',
       '被MLAQ选中', '被VDN选中', '数值状态']


def _clean(t):
    """去掉首尾换行；单元格内保留原换行（excel 里用 wrap_text 显示）"""
    return t.strip()


class QConsistencyRecorder:
    """Reuse the primary result; run only the counterpart on a detached tree.

    observe_compute is called once after each production computation. get_best
    hooks never recompute Q/UCB, and receive the action actually returned.
    The two shadow slots follow the lifetime of the permanent/temporary trees.
    Shadow VDN tables survive case boundaries, but not replacement of a tree.
    """

    def __init__(self, gamma=GAMMA, tol_tie=TOL_TIE, out_name='q_consistency.xlsx'):
        self.gamma, self.tol_tie, self.out_name = gamma, tol_tie, out_name
        self.case_dir = self.target_state = None
        self.rows, self.state_id, self.n_snap = [], {}, 0
        self._pairs = {}

    def start_case(self, case_dir, target_state=None):
        self.case_dir, self.target_state = case_dir, target_state
        self.rows, self.state_id, self.n_snap = [], {}, 0
        # Do not reset shadow Q here: production permanent Q is warm-started too.

    def _sid(self, state):
        if state not in self.state_id:
            self.state_id[state] = len(self.state_id)
        return self.state_id[state]

    def _slot(self, owner):
        return 'temporary' if owner.if_temp_tree else 'permanent'

    @staticmethod
    def _values(tree):
        return {n.state: (n.value, getattr(n, 'ucb_value', None)) for n in tree.nodes}

    @staticmethod
    def _edges(tree):
        return {(n.state, c['action'], c['child'].state)
                for n in tree.nodes for c in n.children}

    def observe_compute(self, tree, algo, target_state, *, owner=None,
                        ucb=False, loop=20):
        """Only the copy calls compute_*; original q_time is never touched."""
        owner = tree if owner is None else owner
        slot = self._slot(owner)
        previous = self._pairs.get(slot)
        if previous is not None and previous['owner'] is not owner:
            previous = None
        # A failed computation must not leave an apparently valid old pair.
        self._pairs.pop(slot, None)
        shadow = copy.deepcopy(tree)
        shadow.q_time = {'total': 0.0, 'calls': 0}
        if algo == 'mlaq':
            shadow.q_tables = (copy.deepcopy(previous['shadow'].q_tables)
                               if previous else [dict() for _ in tree.agent_names])
            if ucb:
                shadow.vdn_compute_ucb_values(target_state)
            else:
                shadow.vdn_compute_q_values(target_state)
        elif algo == 'vdn-mlaq':
            if ucb:
                # Preserve the counterpart's intermediate V on existing inner
                # nodes; new nodes and externally assigned terminal values come
                # from the current production tree.
                if previous:
                    before = self._values(previous['shadow'])
                    for n in shadow.nodes:
                        if n.children and n.state in before:
                            n.value = before[n.state][0]
                shadow.compute_ucb_values(loop=loop)
            elif shadow.get_node(target_state) is not None:
                shadow.compute_values(target_state, loop=loop)
            else:
                # Production MLAQ compute_values raises without a target.
                # Mark unavailable rather than inventing zero reference values.
                for n in shadow.nodes:
                    n.value = None
        else:
            raise ValueError('Unsupported algorithm: ' + algo)
        self.target_state = target_state
        self._pairs[slot] = dict(
            owner=owner, algo=algo, shadow=shadow,
            primary_values=self._values(tree), edges=self._edges(tree))

    def _pair(self, tree, algo, owner=None):
        owner = tree if owner is None else owner
        pair = self._pairs.get(self._slot(owner))
        if pair is None or pair['owner'] is not owner or pair['algo'] != algo:
            raise RuntimeError('No comparison computation for this tree yet')
        return pair

    def _views(self, tree, algo, owner=None, case_end=False):
        pair = self._pair(tree, algo, owner)
        # Detached views permit parser caches and get_best internals to mutate
        # without touching production. Q/VDN tables are not recomputed.
        primary = copy.deepcopy(tree)
        shadow = copy.deepcopy(tree)
        old_shadow = pair['shadow']
        old_values = self._values(old_shadow)
        for n in shadow.nodes:
            n.value, n.ucb_value = old_values.get(n.state, (None, None))
        shadow.q_tables = copy.deepcopy(old_shadow.q_tables)
        if case_end:
            # MLAQ's interaction calculation happens on a disposable copy.
            # Use the captured result instead of the permanent tree's stale V.
            for n in primary.nodes:
                n.value, n.ucb_value = pair['primary_values'].get(n.state, (None, None))
        return ((primary, shadow, pair) if algo == 'mlaq'
                else (shadow, primary, pair))

    @staticmethod
    def _select(tree, state, vdn, ucb, without_virtual):
        node = tree.get_node(state)
        children = [c for c in node.children
                    if not (without_virtual and c['action'] == 'action_virtual')]
        if not children:
            return None
        if not vdn and any((getattr(c['child'], 'ucb_value', None) if ucb
                            else c['child'].value) is None for c in children):
            return None
        if vdn and ucb and any(getattr(c['child'], 'ucb_value', None) is None
                               for c in children):
            return None
        if vdn:
            return tree.vdn_get_best(state, with_ucb=ucb,
                                     without_virtual=without_virtual)[0]
        # Nodes.get_best unconditionally pops action_virtual when requested.
        exclude = without_virtual and any(c['action'] == 'action_virtual'
                                          for c in node.children)
        return node.get_best(with_ucb=ucb, without_virtual=exclude)[0]

    def record_get_best(self, tree, state, phase, site, *, algo,
                        actual_action, owner=None, ucb=False, without_virtual=False):
        mlaq, vdn, pair = self._views(tree, algo, owner)
        state = tree.get_node(state).state  # canonical key, not raw LLM text
        if algo == 'mlaq':
            selected_m = actual_action
            selected_v = self._select(vdn, state, True, ucb, without_virtual)
        else:
            selected_m = self._select(mlaq, state, False, ucb, without_virtual)
            selected_v = actual_action
        snap = f'get_best_{self.n_snap}'
        self.n_snap += 1
        self._append(mlaq, vdn, state, pair, snap, phase, site, ucb,
                     without_virtual, selected_m, selected_v)

    def _append(self, mlaq, vdn, state, pair, snap, phase, site, ucb,
                without_virtual=False, selected_m=None, selected_v=None):
        mc = {c['action']: c for c in mlaq.get_node(state).children}
        kids = [c for c in vdn.get_node(state).children
                if not (without_virtual and c['action'] == 'action_virtual')]
        records = []
        for c in kids:
            a, child = c['action'], c['child']
            mchild = mc[a]['child']
            fresh = (state, a, child.state) in pair['edges']
            pa = vdn._parse_joint(a) or {}
            qi = {ag: vdn.q_tables[i].get((state, pa[ag]), 0.0)
                  for i, ag in enumerate(vdn.agent_names) if ag in pa}
            qm = self.gamma * mchild.value if mchild.value is not None else None
            qv = vdn.get_q_tot(state, a)
            # Same fallback as production vdn_compute_ucb_values; no fictional Qi.
            if qv is None and ucb:
                qv = child.value
            um = getattr(mchild, 'ucb_value', None) if ucb else None
            uv = getattr(child, 'ucb_value', None) if ucb else None
            if not fresh:
                qm = qv = um = uv = None
                qi = {}
            sm = um if ucb else qm
            sv = uv if ucb else qv
            scaled = qv * self.gamma if qv is not None else None
            delta = scaled - qm if scaled is not None and qm is not None else None
            status = ('最后一次计算后新增／修改，尚未计算' if not fresh else
                      'MLAQ目标缺失，无法计算' if qm is None else
                      '非个体动作边' if not pa else '')
            row = [snap, phase, site, 'UCB' if ucb else 'Q',
                   self._sid(state), _clean(state), _clean(a),
                   pa.get('Alice', ''), pa.get('Bob', ''), pa.get('Chad', ''),
                   _clean(child.state), qm, qv,
                   qi.get('Alice'), qi.get('Bob'), qi.get('Chad'), scaled, delta,
                   um, uv, '是' if a == 'action_virtual' else '',
                   '', '', '是' if a == selected_m else '',
                   '是' if a == selected_v else '', status]
            records.append((row, sm, sv))
        for idx in (1, 2):
            scores = [r[idx] for r in records if r[idx] is not None]
            best = max(scores) if scores else None
            for r in records:
                if best is not None and r[idx] is not None and r[idx] >= best - self.tol_tie:
                    r[0][20 + idx] = '是'
        self.rows.extend(r[0] for r in records)

    def record_case_end(self, tree, *, algo):
        # Snapshot, not another training step. Selected-action marks stay blank:
        # a case-end dump did not actually execute get_best.
        pair = self._pairs.get(self._slot(tree))
        if pair is None or pair['owner'] is not tree or pair['algo'] != algo:
            # An unsuccessful case may never have computed permanent-tree Q.
            # Still export its edges, explicitly unavailable instead of zero.
            self._pairs[self._slot(tree)] = dict(
                owner=tree, algo=algo, shadow=copy.deepcopy(tree),
                primary_values={}, edges=set())
        mlaq, vdn, pair = self._views(tree, algo, case_end=True)
        for n in tree.nodes:
            if n.children:
                self._append(mlaq, vdn, n.state, pair, 'case_end',
                             '-', 'case_end', False, without_virtual=True)

    # ==================== 落盘 ====================
    def finish_case(self, tree):
        """写 <case_dir>/q_consistency.xlsx（openpyxl 缺失时退化为同名 .csv）"""
        if not self.rows or not self.case_dir:
            return None
        os.makedirs(self.case_dir, exist_ok=True)
        xlsx = os.path.join(self.case_dir, self.out_name)
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, Alignment, PatternFill
        except ImportError:
            csv_path = xlsx[:-5] + '.csv'
            import csv as _csv
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                w = _csv.writer(f)
                w.writerow(HDR)
                w.writerows(self.rows)
            return csv_path

        wb = Workbook()
        ws = wb.active
        ws.title = 'q_consistency'
        ws.append(HDR)
        for r in self.rows:
            ws.append(r)
        for c in ws[1]:
            c.font = Font(bold=True)
            c.fill = PatternFill('solid', fgColor='DDEBF7')
            c.alignment = Alignment(vertical='center', horizontal='center', wrap_text=True)
        ws.freeze_panes = 'A2'
        widths = [11, 9, 22, 6, 8, 38, 38, 24, 22, 24, 38,
                  13, 13, 13, 13, 13, 13, 15, 13, 13, 11, 12, 11, 11, 10]
        widths.append(28)  # 数值状态
        for i, wdt in enumerate(widths):
            ws.column_dimensions[chr(65 + i)].width = wdt
        wrap_cols = {5, 6, 10}                      # 当前状态 / 联合动作 / 下一状态
        for row in ws.iter_rows(min_row=2):
            for j, cell in enumerate(row):
                cell.alignment = Alignment(vertical='top', wrap_text=(j in wrap_cols))
                if j in (11, 12, 13, 14, 15, 16, 18, 19) and cell.value is not None:
                    cell.number_format = '0.000000000'
                elif j == 17 and cell.value is not None:
                    cell.number_format = '0.00E+00'
        try:
            wb.save(xlsx)
        except PermissionError:
            xlsx = os.path.join(self.case_dir, self.out_name.replace('.xlsx', '_new.xlsx'))
            wb.save(xlsx)
        return xlsx
