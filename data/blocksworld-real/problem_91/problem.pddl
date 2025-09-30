(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        red - block
        blue - block
        green - block
        orange - block
        yellow - block
        purple - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable red)
        (clear red)
        (ontable blue)
        (on green blue)
        (clear green)
        (ontable orange)
        (on yellow orange)
        (clear yellow)
        (ontable purple)
        (clear purple)
        ; red
        ; blue -> green
        ; orange -> yellow
        ; purple
    )
    (:goal (and (on green blue)
        (on yellow green)
        (on red yellow)
        (on purple red)
        (on orange purple)))
)
    