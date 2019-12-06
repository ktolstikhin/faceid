import re
from subprocess import Popen, TimeoutExpired, PIPE


class VideoDeviceSettings:

    def __init__(self, dev='/dev/video0', timeout=None):
        self.dev = dev
        self.timeout = timeout

    def reset_to_defaults(self):
        settings = self.get()

        for entry in settings:

            try:
                entry['value'] = entry['default']
            except KeyError:
                pass

        self.set(settings)

    def get(self):
        set_str = self._exec_shell([
            'v4l2-ctl', '-d', self.dev, '-L'
        ])

        return self._str_to_list(set_str)

    def set(self, settings):
        bool_sets = []
        menu_sets = []
        other_sets = []

        for s_entry in settings:

            if s_entry.get('flags') == 'inactive':
                continue

            s_type = s_entry.get('type')

            if s_type == 'bool':
                bool_sets.append(s_entry)
            elif s_type == 'menu':
                menu_sets.append(s_entry)
            else:
                other_sets.append(s_entry)

        # The order does matter. First apply bool settings, then menu settings,
        # and finally the other ones:
        for vals in (bool_sets, menu_sets, other_sets):

            if not vals:
                continue

            set_ctrl = self._vals_to_str(vals)
            self._exec_shell([
                'v4l2-ctl', '-d', self.dev, '--set-ctrl', set_ctrl
            ])

    def exposure_manual(self):
        # This is a hack. It turns the exposure into a manual mode:
        for val in ('3', '1'):
            self.set([{'name': 'exposure_auto', 'value': val}])

    def _exec_shell(self, args):
        proc = Popen(args, stdout=PIPE, stderr=PIPE)

        try:
            outs, errs = proc.communicate(timeout=self.timeout)
        except TimeoutExpired:
            proc.kill()
            raise

        if errs:
            raise RuntimeError(errs.decode('utf-8'))

        return outs.decode('utf-8') or None

    def _str_to_list(self, set_str):
        set_list = []
        lines = [i.strip() for i in set_str.strip().split('\n')]

        for line in lines:
            # Check if menu entry:
            menu_entry = re.findall(r'^(\d+):\s(.+)$', line)

            if menu_entry:
                val, desc = menu_entry[0]
                menu = set_list[-1].get('menu', {})
                menu[int(val)] = desc
                set_list[-1]['menu'] = menu
            else:
                # Find parameter name and type (int, bool, or menu):
                name_type = re.findall(r'^(\w+)\s*.*\s\((\w+)\)', line)

                if not name_type:
                    continue

                name, param_type = name_type[0]
                param_entry = {'name': name, 'type': param_type}
                params = re.findall(r'(\w+)=(\S+)', line)

                for name, val in params:

                    try:
                        param_entry[name] = int(val)
                    except ValueError:
                        param_entry[name] = val

                set_list.append(param_entry)

        return sorted(set_list, key=lambda x: x['name'])

    def _vals_to_str(self, settings):
        toks = [f'{s["name"]}={s["value"]}' for s in settings]

        return ','.join(toks)

