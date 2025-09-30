(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        purple - block
        orange - block
        yellow - block
        red - block
        green - block
        blue - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable purple)
        (on orange purple)
        (on yellow orange)
        (on red yellow)
        (on green red)
        (clear green)
        (ontable blue)
        (clear blue)
        ; purple -> orange -> yellow -> red -> green
        ; blue
    )
    (:goal (and (on yellow orange)
        (on red yellow)
        (on green red)
        (on purple green)
        (on blue purple)))
)
    